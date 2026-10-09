import torch
from torch import nn
import torch.nn.functional as F
import math
import pdb
from functools import partial


# Diffusion 모델의 파이프라인 전체를 관리하는 container
class Diffusion(nn.Module):
    def __init__(self,
        timesteps = 200,
        beta_start = 0.1,
        beta_end = 0.1,
        hyper_w = 0.1,
        fusion_mode = 'clsattn',
        max_len = 50,
        p = 0.6 # 수정: CFG dropout 확률을 인자로 받도록 추가
                    ):
        super().__init__()

        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = hyper_w
        self.p = p # 수정: args.p 저장 (cacu_h의 CFG dropout에 사용)
        
        self.fusion_mode = fusion_mode
        
        self.model = ConditionNet(fusion_mode = fusion_mode, max_len = max_len ,state_size = max_len)
        
        # 각 타임스텝 t에서 추가할 노이즈의 분산 β_t
        self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # 알파 누적곱 ᾱ_t
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) # ᾱ_(t-1). [1.0, ᾱ_1, ᾱ_2, ..., ᾱ_{T-1}]
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 학습에 쓰이는 상수
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 추론에 쓰이는 상수
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    # 순방향 노이즈 추가. Reparameterization 덕분에 타임스텝 t를 하나씩 거치지 않고, 한 번의 실행으로 완전한 노이즈가 섞인 샘플을 만들 수 있다.
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise # e_n t
        
        
    # 학습 (손실 계산)
    def p_losses(self, seq,len_seq,target,content=None, noise=None, loss_type="l2", authenticity= None):

        content_features, mask = self.model.content_encoder(content) # 후보 뉴스 시퀀스 인코딩
        seq = torch.stack(seq,dim=0)

        x_start = self.model.cacu_x(target) # 타겟 뉴스 임베딩. Diffusion 원본인 e_n 0
        h,state_hidden =  self.model.cacu_h(seq, len_seq, self.p, mask[:,1:]) # 사용자 서치 이력 인코딩 # 수정: 0.6 하드코딩 → args.p 사용

        t = torch.randint(0, self.timesteps, (len_seq.shape[0], ), device=len_seq.device).long() # 각 샘플마다 무작위 타임스텝 t 선택

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) # e_n 0에 t단계 노이즈를 한 번에 추가. e_n t

        authenticity = self.model.content_reflect(authenticity) # e_real을 128-dim으로 투영

        # ConditionNet.forward()를 호출해서 노이즈 예측
        predicted_noise = self.model(x_noisy, h, t, content_features, state_hidden, authenticity, mask)

        # e_n t와 예측된 노이즈로 e_n 0 역산하기
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_noisy.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
        predicted_x0 = (x_noisy - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t  # e_n t 공식으로 e_n 0 공식 유도

        train_emb = self.model.news_embeddings.weight # 전체 뉴스 임베딩 테이블
        encoded_x_list = []
        for i in range(0,train_emb.size(0),seq.size(0)):
            encoded_emb = self.model.content_reflect(train_emb[i:i + seq.size(0)]) # 128-dim으로 투영
            encoded_x_list.append(encoded_emb) 

        encoded_x = torch.cat(encoded_x_list, dim=0)
        # 수정: 정규화 내적(cosine). 이방성 임베딩에서 순수 내적은 소수 허브 아이템만 반환(아이템이 자기 자신도 못 찾음)
        # → 학습 CE가 모든 예측을 허브 방향으로 끌어 모드 붕괴 유발. 정규화가 논문 "inner product" 수치의 전제.
        tau = 0.07  # CE를 날카롭게 하는 온도 (InfoNCE 방식)
        scores = torch.matmul(F.normalize(predicted_x0, dim=-1), F.normalize(encoded_x, dim=-1).transpose(0, 1)) / tau # 추천 스코어

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise) # L_d. 실제 노이즈와 예측 노이즈 차이 비교
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss, scores

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    # 역방향 1 step
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index,content_features,hidden_state,authenticity_0, authenticity_1,attn_mask):

        # Negative prompt 공식
        predicted_noise = (1 + self.w) * model_forward(x, h, t, content_features, hidden_state, authenticity_0, attn_mask) \
                          - self.w * model_forward_uncon(x, t, authenticity_1)

        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        x_start = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t # e_n t로부터 e_n 0 추정

        # 사후 평균 계산
        model_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )

        if t_index == 0: # 마지막 스텝에선 노이즈 없이 평균만 반환
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise # 중간 스텝은 노이즈 추가
        
    # 전체 추론 파이프라인
    @torch.no_grad()
    def sample(self, seq, len_seq, target,content=None,authenticity_0=None, authenticity_1= None):
        content_features,attn_mask = self.model.content_encoder(content)
        seq = torch.stack(seq,dim=0)
        
        h,hidden_state =  self.model.predict(seq, len_seq,attn_mask[:,1:])
        if h.dim()==1:
            h = h.unsqueeze(0)
        
        x = torch.randn_like(h) # 순수 가우시안 노이즈인 e_n T에서 시작


        
        authenticity_0 = self.model.content_reflect(authenticity_0) # e_real을 128-dim으로 투영. ((e_fake는 forward_uncon 내부에서 처리)
        for n in reversed(range(0, self.timesteps)): # T -> 0으로 역방향 반복
            x = self.p_sample(self.model.forward, self.model.forward_uncon, x, h, torch.full((h.shape[0], ), n, device=len_seq.device, dtype=torch.long), n,content_features, hidden_state,authenticity_0, authenticity_1,attn_mask)
        # 루프 완료 후 x는 e_n 0 (생성된 사용자 관심사 임베딩)

        test_emb = self.model.news_embeddings.weight # 전체 뉴스 임베딩
        
        mask = torch.ones(test_emb.size(0), device=test_emb.device, dtype=torch.bool)
        mask[0] = False 
        
        encoded_x_list = []
        for i in range(0,test_emb.size(0),seq.size(0)):
            encoded_emb = self.model.content_reflect(test_emb[i:i + seq.size(0)])
            encoded_x_list.append(encoded_emb)
        
        encoded_x = torch.cat(encoded_x_list, dim=0)
        # 수정: 학습 손실과 동일하게 정규화 내적(cosine)으로 검색 (topk는 온도에 불변이라 τ 생략)
        scores = torch.matmul(F.normalize(x, dim=-1), F.normalize(encoded_x, dim=-1).transpose(0, 1)) # 추천 스코어 계산

        scores = scores.masked_fill(~mask, float('-inf'))

        return x, scores


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):

    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    alpha_bar = (torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2)
    alpha_bar = alpha_bar / alpha_bar[0] 
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clip(beta, 0.0001, 0.9999)  

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist

def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
    


# Phase 2 추천 확산 모델 (Recommended Diffusion Model)의 Conditional Denoiser
class ConditionNet(nn.Module):
    def __init__(self, hidden_size=128, state_size=50, dropout=0.1, fusion_mode='clsattn', device='cuda:0', num_heads=4,max_len = 50):
        super(ConditionNet, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fusion_mode = fusion_mode
        self.device = device
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # BERT 768-dim 뉴스 임베딩을 128-dim으로 압축
        self.content_reflect = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LeakyReLU(),
            norm_layer(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        ) 
        
        self.content_pos = class_token_pos_embed(hidden_size, max_len)
        self.content_embed = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        

        # 빈 조건 벡터 phi
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.ln_4 = nn.LayerNorm(hidden_size)
        self.ln_5 = nn.LayerNorm(hidden_size)
        
        # Attention 레이어
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.mh_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.cross_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.cross_attn_2 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        
        
        self.nn_1 = nn.Linear(hidden_size, hidden_size)
        self.nn_2 = nn.Linear(2*hidden_size,hidden_size)
        self.nn_3 = nn.Linear(hidden_size,hidden_size)
        self.content_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            norm_layer(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 768)
        )
        
        
        
        
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )



        self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*4, self.hidden_size)
        )
        self.init_weights()
        
    def init_weights(self):

        for layer in self.content_reflect:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    # Conditional Denoising
    def forward(self, x, h, step, content=None, state_hidden=None, authenticity=None, mask=None):

        content = self.nn_1(content) # C_k
        authenticity = self.nn_1(authenticity) # e_real
        
        cosine_sim = F.cosine_similarity(content, authenticity.unsqueeze(1), dim=-1)
        gate = (1 + torch.tanh(cosine_sim)) / 2 # 유사도 기반 가중치 A_k

        # C_k adj 계산
        content = content * gate.unsqueeze(-1) + authenticity.unsqueeze(1) * (1 - gate).unsqueeze(-1)
        
        # time step 인코딩
        t = self.step_mlp(step)
        x = x.squeeze(1)
        
        if self.fusion_mode == 'clsattn':
            x_t = self.nn_2(torch.cat((x, t), dim=1))
            x_t = x_t.unsqueeze(1)
            cls_token = content[:, 0].unsqueeze(1) # cls 토큰만 사용하는 모드
            e_hat = self.cross_attn_1(x_t, cls_token, cls_token)
            c_hat = self.cross_attn_2(e_hat, h.unsqueeze(1), h.unsqueeze(1))
            res = self.feed_forward(c_hat)
            res = res.squeeze(1)
        # 논문에 나온 모드
        elif self.fusion_mode == 'seqattn': 
            x_t = self.nn_2(torch.cat((x, t), dim=1))
            x_t = x_t.unsqueeze(1)

            e_hat = self.cross_attn_1(x_t, content, content, mask)
            c_hat = self.cross_attn_2(e_hat, state_hidden, state_hidden, mask[:,1:]) 
            h_hat = self.mh_attn_1(c_hat, h.unsqueeze(1), h.unsqueeze(1))
            res = self.feed_forward(h_hat)
            res = res.squeeze(1)
        else:
            res = self.diffuser(torch.cat((x, h, t, content[:, 0, :]), dim=1))

        return res

    # Unconditional Denoising. Inference 단계에서 사용 (Diffusion class의 p_sample 함수에서 사용됨)
    def forward_uncon(self, x, step, authenticity):
        # 사용자 이력 h 대신 빈 조건 φ 사용
        phi = self.none_embedding(torch.tensor([0]).to(self.device))
        phi = torch.cat([phi.view(1, self.hidden_size)]*x.shape[0], dim=0)
        t = self.step_mlp(step)
        if authenticity is not None:
            if self.fusion_mode == 'clsattn':
                authenticity = self.content_reflect(authenticity).repeat(x.shape[0], 1)
                authenticity = self.nn_1(authenticity)
                x_t = self.nn_2(torch.cat((x,t), dim=1))
                x_t = x_t.unsqueeze(1)
                auth_kv = authenticity.unsqueeze(1)
                phi_kv = phi.unsqueeze(1)
                e_hat = self.cross_attn_1(x_t, auth_kv, auth_kv)
                c_hat = self.cross_attn_2(e_hat, phi_kv, phi_kv)
                res = self.feed_forward(c_hat)
                res = res.squeeze(1)
            elif self.fusion_mode == 'seqattn':
                authenticity = self.content_reflect(authenticity).repeat(x.shape[0], 1)
                authenticity = self.nn_1(authenticity)
                x_t = self.nn_2(torch.cat((x, t), dim=1))
                x_t = x_t.unsqueeze(1)
                auth_kv = authenticity.unsqueeze(1)
                phi_kv = phi.unsqueeze(1)
                e_hat = self.cross_attn_1(x_t, auth_kv, auth_kv)
                c_hat = self.cross_attn_2(e_hat, phi_kv, phi_kv)
                h_hat = self.mh_attn_1(c_hat, phi_kv, phi_kv)
                res = self.feed_forward(h_hat)
                res = res.squeeze(1)
            else:
                res = self.diffuser(torch.cat((x, phi, t, authenticity), dim=1))



        else:
            res = self.diffuser(torch.cat((x, phi, t, phi), dim=1))

        return res

    def cacu_x(self, x):
        x = self.content_reflect(self.news_embeddings(x.long()))
        x = x.squeeze(1)
        
        return x
    
    # 후보 뉴스 인코딩
    def content_encoder(self, content):
        text, mask = content
        text = self.content_reflect(text)
        clsheadmask = torch.ones(mask.shape[0], 1, dtype=torch.bool).to(mask.device)
        try:
            mask = torch.cat([clsheadmask, mask],dim=-1)
        except:
            pdb.set_trace()
        

        text = self.content_pos(text)
        content_feature = self.content_embed(text, text, text, mask)
        

        return content_feature, mask

    # 사용자 이력 인코딩 (학습용)
    def cacu_h(self, states, len_states, p ,attn_mask):
        # 임베딩, 위치 인코딩
        inputs_emb = self.content_reflect(self.news_embeddings(states))
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device)) # 위치 정보 추가
        seq = self.emb_dropout(inputs_emb)

        # 패딩 마스킹
        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)
        seq *= mask

        # Self-Attention. 시퀀스 표현
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq,seq, attn_mask)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)

        # 실제 토큰 추출
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        # 학습때만 dropout 사용. predict() 단계에는 없음.
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)


        return h, ff_out
    
    # 사용자 이력 인코딩 (추론용)
    def predict(self, states, len_states,attn_mask):
        inputs_emb = self.content_reflect(self.news_embeddings(states))
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)

        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)
        seq *= mask

        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq,seq,attn_mask)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)

        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()
        
        return h, ff_out
    
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class Attention(nn.Module):
    def __init__(self, attention_dropout=0.5):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.matmul(q, k.transpose(-2, -1))

        if scale is not None:  # 수정: scaled dot-product attention — scale 미적용 버그 수정
            attention = attention * scale



        if attn_mask is not None:
            if attention.shape[2] == attention.shape[3]: # self attention
                attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)
                attention = attention.masked_fill_(~attn_mask.unsqueeze(1), float(-1e20))
            else:
                attention = attention.masked_fill_(~attn_mask.unsqueeze(1).unsqueeze(1), float(-1e20))
            
                
        
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)


        return attention   
    
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=128, out_dim=128, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention =Attention(dropout)
        self.linear_final = nn.Linear(model_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        B1, N1, C1 = query.shape
        B2, N2, C2 = key.shape   
        residual = query
        
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.reshape(B2, N2, num_heads, dim_per_head)
        value = value.reshape(B2, N2, num_heads, dim_per_head) 
        query = query.reshape(B1, N1, num_heads, dim_per_head)

        
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        scale = (self.dim_per_head)**-0.5
        attention = self.dot_product_attention(query, key, value, 
                                               scale, attn_mask)

        attention = attention.transpose(1, 2)
        attention = attention.reshape(B1, N1, C1)
        

        output = self.linear_final(attention)

        output = self.dropout(output)

        output = self.layer_norm(residual + output)

        
        return output    
    
    
    


class class_token_pos_embed(nn.Module):
    def __init__(self, embed_dim,num_tokens):
        super(class_token_pos_embed, self).__init__()
        
        self.num_tokens = 1  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens+self.num_tokens, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):  

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        return x



# 학습 가능한 인코더
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # 함수 안에 학습할 파라미터 w와 b가 들어있다.
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


# Phase 1 진실성 특징 분리기 (Disentangling Authenticity Features)
class ModelWithEmbeddingIB(nn.Module): # (Information Bottleneck)
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, num_classes, n_iters=100, logit_scale=1.5, dropout_rate=0.3):
        super(ModelWithEmbeddingIB, self).__init__()
        self.n_iters = n_iters
        self.logit_scale = logit_scale
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

        self.encoder_r = Encoder(input_dim, hidden_dim, bottleneck_dim) # 진실성 특징 관련 z_r 생성: 768 → 128 → 128
        self.encoder_i = Encoder(input_dim, hidden_dim, bottleneck_dim) # 진실성 특징 무관 z_i 생성: 768 → 128 → 128

        # num_classes = 2 (real, fake)
        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=input_dim) # 클래스 인덱스(0,1)를 768-dim 벡터로 변환

        self.decoder = nn.Sequential(
            nn.Linear(2 * bottleneck_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(hidden_dim, input_dim)
        ) # z_r + z_i로 원본 뉴스 임베딩 복원 (L_r에 사용)
        
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),  
            nn.Linear(hidden_dim, num_classes)
        ) # z_r만으로 real/fake 분류 (L_cls에 사용)
        
    def cost_matrix_cosine(self, x, y, eps=1e-5):
        assert x.dim() == y.dim() == 2
        assert x.size(-1) == y.size(-1)
        x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
        y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
        cosine_sim = torch.matmul(x_norm, y_norm.transpose(-2, -1))
        cosine_dist = 1 - cosine_sim # 코사인 비유사도. 두 벡터 방향이 같을수록 0에 가까워짐
        return cosine_dist.unsqueeze(0)

    def trace(self, x):
        b, m, n = x.size()
        assert m == n
        mask = torch.eye(n, dtype=torch.uint8, device=x.device
                        ).unsqueeze(0).expand_as(x).bool()
        trace = x.masked_select(mask).contiguous().view(
            b, n).sum(dim=-1, keepdim=False)
        return trace

    # Sinkhorn-Knopp 알고리즘
    @torch.no_grad()
    def ipot(self, C, beta, iteration, k, eps=1e-8): # 수정: 2-프로토타입 등 불균형 marginal에서 0 나눗셈 발산 방지용 eps 추가
        b, m, n = C.size()
        sigma = torch.ones(b, m, dtype=C.dtype, device=C.device)
        T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
        A = torch.exp(-C.transpose(1, 2) / beta) # Gibbs 커널 K = exp(-j / beta)

        for _ in range(iteration):
            Q = A * T
            sigma = sigma.view(b, m, 1)
            for _ in range(k):
                delta = 1 / (Q.matmul(sigma).view(b, 1, n) + eps) # a 업데이트 # 수정: 분모 eps
                sigma = 1 / (delta.matmul(Q) + eps) # b 업데이트 # 수정: 분모 eps
            T = delta.view(b, n, 1) * Q * sigma # transport matrix T
        return T

    def optimal_transport_dist(self, z_label, z_r, cost=None, beta=0.5, iteration=50, k=1):
        if cost is None:
            cost = self.cost_matrix_cosine(z_label, z_r) # J: 코사인 비유사도 행렬

        T = self.ipot(cost.detach(), beta, iteration, k) # T
        distance = self.trace(cost.matmul(T.detach())) # trace(J @ T)
        return distance.mean() 

    def contrastive_loss_per_label(self, z_label, z_r, z_i, labels):
        unique_labels = torch.unique(labels)
        total_loss = 0.0

        # label별로 독립적으로 계산. z_l과 z_r이 가까워지도록, z_l과 z_i가 멀어지도록
        for label in unique_labels:

            indices = (labels == label).nonzero().squeeze()
            
            if indices.dim() == 0:
                indices = indices.unsqueeze(0) 

            z_label_cur = z_label[indices]
            z_r_cur = z_r[indices]
            z_i_cur = z_i[indices]

            positive_samples = z_r_cur.unsqueeze(1)
            negative_samples = z_i_cur.unsqueeze(1)

            samples = torch.cat([positive_samples, negative_samples], dim=1) # z_r + z_i = z
            samples = F.normalize(samples, dim=-1)
            z_label_cur = F.normalize(z_label_cur, dim=-1)

            logits = self.logit_scale * F.cosine_similarity(z_label_cur.unsqueeze(1), samples, dim=2) # cosine_similarity(z_label, [z_r, z_i])
            bz, K, _ = positive_samples.shape
            target = torch.cat([torch.ones(bz, K), torch.zeros(bz, K)], dim=1).to(z_label_cur.device) # [1, 0]
            logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)  

            loss = self.kl_div_loss(logits, target) # D_KL(log_softmax(logits), target)
            total_loss += loss

        return total_loss / len(unique_labels)

    # 메인 흐름
    def forward(self, x, labels):
        z_r = self.encoder_r(x) # z_r
        z_i = self.encoder_i(x) # z_i

        label_embed = self.label_embedding(labels) # E_label (샘플별 라벨 임베딩. contrastive/재구성에 사용)
        z_label = self.encoder_r(label_embed) # z_l

        # 수정: OT를 논문(§4.1)의 2개 클래스 프로토타입 {z_real, z_fake} 앵커에 정렬 (기존: 배치×배치)
        prototype_ids = torch.arange(self.label_embedding.num_embeddings, device=x.device) # [0(real), 1(fake)]
        z_prototypes = self.encoder_r(self.label_embedding(prototype_ids)) # 수정: [2, dim] 클래스 프로토타입 z_l
        ot_distance = self.optimal_transport_dist(z_prototypes, z_r) # 2) d_OT(z_r, z_l): cost=[2, batch] # 수정
        contrastive_loss_zr_zlabel = self.contrastive_loss_per_label(z_label, z_r, z_i, labels) # 3) L_c

        # 4) L_r 계산을 위한 값들
        x_recon = []
        x_ori = []
        for label in torch.unique(labels):
            indices = (labels == label).nonzero().squeeze()
            
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)
                
            z_r_shuffled = z_r[indices[torch.randperm(indices.size(0))]]
            z_i_ori = z_i[indices]
            x_recon_label = self.decoder(torch.cat([z_r_shuffled, z_i_ori], dim=1))
            x_recon.append(x_recon_label)
            x_ori.append(x[indices])
        x_recon = torch.cat(x_recon, dim=0)
        x_ori = torch.cat(x_ori, dim=0)
        
        logits = self.classifier(z_r) # z_r만으로 real/fake 분류
        
        return x_recon, z_label, z_r, z_i, ot_distance, x_ori, contrastive_loss_zr_zlabel, logits
