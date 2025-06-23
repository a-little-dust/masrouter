from typing import List, Dict, Optional
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import gammaln
import math

from MAR.LLM.llm_embedding import SentenceEncoder
from MAR.Graph.graph import Graph
from MAR.Utils.utils import get_kwargs, plot_embedding_heatmap, plot_row_similarity
from MAR.Utils.globals import Cost
from loguru import logger

class GFusion(nn.Module):
    def __init__(self, d_model:int=384):
        """
        Graph Fusion Module: GFM
        Input: x: [xx, d], y: [yy, d]
        Output: z: [xx, d]
        """
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    # 基于注意力机制，最终输出融合了原始信息和注意力信息
    def forward(self, x, y):
        Q = self.query_proj(x)      # [xx, d]
        K = self.key_proj(y)        # [yy, d]
        V = self.value_proj(y)      # [yy, d]

        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / (Q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [xx, yy]
        context = torch.matmul(attn_weights, V)  # context: [xx, d]
        context = F.normalize(context, p=2, dim=1)
        z = self.out_proj(x + context)  # [xx, d]
        return z

std2 = 0.1
var2 = std2 * std2
log_var2 = math.log(var2)

# 变分自编码器，包含编码器和解码器
class VAE(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=64, latent_dim=64):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)  # μ, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)*std2
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h) # x_hat

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, z, mu, log_var

def vae_loss_function(x_hat, x, mu, log_var):
    MSE = F.mse_loss(x_hat, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 - log_var2 + log_var - (mu.pow(2) + log_var.exp())/var2)
    return MSE + KLD

class MasRouter(nn.Module):
    """
    Input: Text descriptions of queries, tasks, LLMs, collab methods, roles, and corresponding tools
    Output: Task classification, number and types of LLMs required for each query, recommended collab reasoning methods and roles
    Description: LLMs include chatgpt, gemini, llama, etc., collab reasoning methods include single-agent CoT reasoning, multi-agent debate reasoning, multi-agent collaboration reasoning based on certain topological structures, roles include various identities, and various tools can be used, such as python compilers, wiki searches, etc.
    Requirements: Build a trainable model to construct the optimal multi-agent system
    """
    def __init__(self, in_dim:int = 384, hidden_dim:int = 64, max_agent:int = 6, temp:float=0.5, device=None):
        """
        query: N*d tensor, N is the number of queries, d is the dimension of each query
        task: N_t*d tensor, N_t is the number of tasks, d is the dimension of each task
        llm: N_l*d tensor, N_l is the number of llm, d is the dimension of each llm
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder = SentenceEncoder(device=self.device)
        self.task_classifier = TaskClassifier(input_dim = in_dim, hidden_dim=hidden_dim, device=self.device,temp=temp)
        self.collab_determiner = CollabDeterminer(input_dim = in_dim, context_input_dim = in_dim , hidden_dim = hidden_dim,device=self.device,temp=0.8)
        self.num_determiner = NumDeterminer(input_dim = in_dim, hidden_dim=hidden_dim,max_agent=max_agent, device=self.device)
        self.role_allocation = RoleAllocation(input_dim = in_dim, context_input_dim = 2* hidden_dim, hidden_dim=hidden_dim,device=self.device,temp=temp)
        self.llm_router = LLMRouter(device=self.device,max_agent=max_agent,temp=1.0)

    # 核心函数，依次执行：协作方式选择，代理数量确定，角色选择，LLM选择
    def forward(self, queries:List[str], tasks:List[Dict[str, str]], 
                llms: List[Dict[str, str]], collabs:List[Dict[str, str]], given_task: Optional[List[int]] = None, 
                prompt_file:str='MAR/Roles/FinalNode/gsm8k.json'):
        """
        queries:List[Dict[str, str]]: List of queries
        tasks:List[Dict[str, str]]: List of tasks
        llms:List[Dict[str, str]]: List of llms
        collabs:List[Dict[str, str]]: List of collabs
        """
        # Preprocess data
        tasks_list = self._preprocess_data(tasks)
        llms_list = self._preprocess_data(llms)
        collabs_list = self._preprocess_data(collabs)
        task_role_database, task_role_emb = self.encoder_roles() # task_role_database: Dict[str, List[Dict[str, str]]], task_role_emb: Dict[str, torch.Tensor]

        # Text embedding
        queries_embedding = self.text_encoder(queries) # N_q*d tensor, N_q is the number of queries, d is the dimension of each query
        tasks_embedding = self.text_encoder(tasks_list) # N_t*d tensor, N_t is the number of tasks, d is the dimension of each task
        llms_embedding = self.text_encoder(llms_list) # N_l*d tensor, N_l is the number of llms, d is the dimension of each llm
        collabs_embedding = self.text_encoder(collabs_list) # N_r*d tensor, N_r is the number of collabs, d is the dimension of each collab
        
        # Task classification
        selected_tasks_idx, tasks_probs, query_context = self.task_classifier(queries_embedding, tasks_embedding) # N_q, N_q*1，N_q*hidden_dim
        selected_tasks:List[Dict[str,str]] = [tasks[idx] for idx in selected_tasks_idx] if given_task is None else [tasks[idx] for idx in given_task]
        # 根据选中的所有任务，整理出所有角色的列表
        tasks_role_list:List[List[Dict[str,str]]] = [task_role_database[task['Name']] for task in selected_tasks]
        # 根据选中的所有任务，整理出所有角色的嵌入向量张量
        tasks_role_emb_list:List[torch.Tensor] = [task_role_emb[task['Name']] for task in selected_tasks]

        # Collaboration method selection
        # 选中的协作方式 在agent.py中，被转为self.reason，获得相应的reasoning_prompt
        selected_collabs_idx, collab_log_probs, collab_context, collab_vae_loss = self.collab_determiner(collabs_embedding, queries_embedding) # N_q, N_q*1，N_q*hidden_dim
        selected_collabs:List[Dict[str,str]] = [collabs[idx] for idx in selected_collabs_idx]
        
        # Number of agents determination
        agent_num_int, agent_num_float, num_vae_loss = self.num_determiner(queries_embedding) # N_q*1, N_q*1
        
        # Role selection
        selected_roles_idx, role_log_probs, role_context, role_vae_loss = self.role_allocation(tasks_role_emb_list, torch.concat([query_context, collab_context],dim=-1), agent_num_int) # N_q*agent_num, N_q*1，N_q*hidden_dim
        selected_roles:List[List[Dict[str,str]]] = [[tasks_roles[selected_role_id.item()] for selected_role_id in selected_roles_id_list] for tasks_roles, selected_roles_id_list in zip(tasks_role_list, selected_roles_idx)]
        
        # LLM allocation
        selected_llms_idx, llm_log_probs, llm_vae_loss = self.llm_router(llms_embedding, torch.concat([query_context, collab_context, role_context],dim=-1), agent_num_int, agent_num_float) # N_q*1，N_q*hidden_dim
        selected_llms:List[List[Dict[str,str]]] = [[llms[idx] for idx in selected_llms_id_list] for selected_llms_id_list in selected_llms_idx]
        log_probs = llm_log_probs + role_log_probs + collab_log_probs # N_q*1

        vae_loss = collab_vae_loss + num_vae_loss + role_vae_loss + llm_vae_loss

        final_result = []
        costs = []
        for query, task, llms, collab, roles in zip(queries, selected_tasks, selected_llms, selected_collabs, selected_roles):
            previous_cost = Cost.instance().value
            kwargs = get_kwargs(collab['Name'], len(llms))
            llm_names = [llm['Name'] for llm in llms]
            role_names = [role['Name'] for role in roles]
            logger.info(f'Query: {query}')
            logger.info(f'Task: {task["Name"]}')
            logger.info(f'LLMs: {llm_names}')
            logger.info(f'Reasoning: {collab["Name"]}')
            logger.info(f'Roles: {role_names}')
            logger.info('-----------------------------------')
            # 构建graph
            g = Graph(domain = task['Name'], llm_names = llm_names, agent_names = role_names, 
                      decision_method = "FinalRefer", prompt_file = prompt_file, reasoning_name=collab["Name"], **kwargs)
            self.g = g
            # 运行graph，在运行过程中可以优化时间和空间连接（也可以不优化）
            final_result.append(g.run(inputs={"query":query}, num_rounds=kwargs["num_rounds"])[0][0])
            # 计算成本
            costs.append(Cost.instance().value - previous_cost)

        return final_result, costs, log_probs, tasks_probs, vae_loss, agent_num_float
    
    def _preprocess_data(self, raw_data:List[Dict[str, str]]):
        """
        raw_data: List of dictionaries with 'Name' and 'Description' keys
        """
        get_name_description = lambda x: x['Name'] + ' : ' + x['Description']
        return [get_name_description(data) for data in raw_data]
    
    def encoder_roles(self):
        """
        Return:
            task_role_database: Dict[str, List[Dict[str, str]]]: A dictionary of task-role database
            task_role_emb: Dict[str, torch.Tensor]: A dictionary of task-role embeddings. The tensor is N_t_r*d.
            键是任务名称，值是该任务所有角色的嵌入向量张量
        """
        logger.info('Loading role embeddings...')
        task_role_database = {}
        task_role_emb = {}
        path = 'MAR/Roles'
        # 遍历 MAR/Roles 目录的json文件，每个文件代表一个角色
        for task in os.listdir(path):
            task_path = os.path.join(path, task)
            if os.path.isdir(task_path):#处理这个任务的所有角色
                task_role_database[task] = []
                roles_list = []
                for role in os.listdir(task_path):
                    if role.endswith('.json'):
                        role_path = os.path.join(task_path, role)
                        role_profile = json.load(open(role_path, 'r', encoding='utf-8'))
                        task_role_database[task].append(role_profile)#把角色信息添加到这个task对应的列表
                        roles_list.append(json.dumps(role_profile))
                if len(roles_list):#如果这个任务有角色，则计算所有角色的嵌入向量
                    task_role_emb[task] = self.text_encoder(roles_list).to(self.device)
        logger.info('Role embeddings loaded.')
        return task_role_database, task_role_emb

# 任务分类器
class TaskClassifier(nn.Module):
    def __init__(self, input_dim:int=384, hidden_dim:int=64, temp:float = 1.0, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_encoder = nn.Linear(input_dim, hidden_dim)
        self.task_encoder = nn.Linear(input_dim, hidden_dim)
        self.temp = temp

    """
    tasks是task_list编码后的结果，queries是queries编码后的结果
    使用 query_encoder 将查询转换为隐藏表示，使用 task_encoder 将任务转换为隐藏表示
    两个encoder都是线性层，输入是input_dim，输出是hidden_dim
    计算查询和任务之间的相似度，选择最相似的任务
    """
    def forward(self, queries, tasks):
        """
        queries: N_q*d tensor, N_q is the number of queries, d is the dimension of each query
        tasks: N_t*d tensor, N_t is the number of tasks, d is the dimension of each task
        """ 
        query_embedding = self.query_encoder(queries) # N_q*hidden_dim
        task_embedding = self.task_encoder(tasks) # N_t*hidden_dim
        query_embedding = F.normalize(query_embedding, p=2, dim=1) # L2 normalization
        task_embedding = F.normalize(task_embedding, p=2, dim=1) # L2 normalization
        scores = torch.matmul(query_embedding, task_embedding.T) # N_q*N_t
        scores = F.softmax(scores/self.temp, dim=1) # N_q*N_t
        selected_tasks_id = torch.argmax(scores, dim=1) # N_q

        return selected_tasks_id, scores, query_embedding

# 协作方式选择。注意这里用VAE编码
class CollabDeterminer(nn.Module):
    def __init__(self, input_dim=384, context_input_dim=384, hidden_dim=64, temp=1.0, device=None):
        super().__init__()
        self.collab_encoder = VAE(input_dim, hidden_dim, hidden_dim)
        self.context_encoder = VAE(context_input_dim, hidden_dim, hidden_dim)
        # 使用 GFusion 可以将这两种潜在表示进行融合，得到一个融合后的潜在表示
        self.collab_context_encoder = GFusion(d_model=hidden_dim)
        self.temp = temp
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """
    对协作方法（collabs）和上下文（contexts）分别进行VAE编码
    计算上下文和协作方法之间的相似度分数
    """
    def forward(self, collabs:torch.Tensor, contexts:torch.Tensor):
        # 1. 使用VAE编码协作方法
        collab_hat, collab_z, collab_mu, collab_logvar = self.collab_encoder(collabs)  # N_t*latent_dim
        # collab_hat: 重构后的协作方法表示
        # collab_z: 潜在空间中的协作方法表示
        # collab_mu: 均值
        # collab_logvar: 对数方差
        collab_z = F.normalize(collab_z, p=2, dim=1) # 对潜在表示进行L2归一化

        # 2. 使用VAE编码上下文
        context_hat, context_z, context_mu, context_logvar = self.context_encoder(contexts)  # N_q*latent_dim
        context_z = F.normalize(context_z, p=2, dim=1) # 对潜在表示进行L2归一化

        # 3. 计算相似度分数
        scores = torch.matmul(context_z, collab_z.T) # 计算上下文和协作方法之间的相似度
        scores = torch.softmax(scores / self.temp, dim=1) # 使用温度参数进行softmax归一化

        # 4. 计算VAE的损失
        vae_loss1 = vae_loss_function(collab_hat, collabs, collab_mu, collab_logvar) # 协作方法的VAE损失
        vae_loss2 = vae_loss_function(context_hat, contexts, context_mu, context_logvar) # 上下文的VAE损失
        vae_loss = vae_loss1 + vae_loss2 # 总VAE损失

        # 5. 选择协作方法
        scores_cumsum = torch.cumsum(scores, dim=1) # 计算累积概率
        random_num = torch.rand([scores.size(0),1], device=self.device) # 生成随机数
        selected_index = (scores_cumsum > random_num).float().argmax(dim=1) # 基于累积概率选择索引

        # 6. 计算对数概率和获取选中的协作方法嵌入
        log_probs = torch.log(scores[torch.arange(scores.size(0)), selected_index]).unsqueeze(1) # 计算选中项的概率的log。对数转换后，乘法变成加法，更稳定
        collab_embedding = collab_z[selected_index] # 获取选中协作方法的嵌入表示

        # 7. 返回结果
        return selected_index, log_probs, collab_embedding, vae_loss
        # selected_index: 选中的协作方法索引
        # log_probs: 选择的对数概率
        # collab_embedding: 选中协作方法的嵌入表示
        # vae_loss: VAE的总损失

# 代理数量选择。用VAE编码，连接全连接层，获得任务难度，然后根据难度算出代理数量
class NumDeterminer(nn.Module):
    def __init__(self, input_dim:int=384, hidden_dim:int = 64, max_agent:int = 6, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae = VAE(input_dim, hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1) # 全连接层，从hidden_dim -> 1
        self.max_agent = max_agent
        
    def forward(self, queries:torch.Tensor):
        """
        输入: queries - 查询的嵌入向量，形状为 N_q*input_dim
        输出: 
            - agent_num_int: 整数形式的代理数量
            - agent_num_float: 浮点数形式的代理数量
            - vae_loss: VAE的损失值
        """
        # 1. 使用VAE编码查询
        x_hat, z, mu, log_var = self.vae(queries)
        # x_hat: 重构后的查询表示
        # z: 潜在空间中的查询表示
        # mu: 均值
        # log_var: 对数方差
        
        # 2. 对潜在表示进行L2归一化，使向量长度为1
        z = F.normalize(z, p=2, dim=1)
        
        # 3. 通过全连接层预测查询难度
        query_difficulty = self.fc(z)  # 将潜在表示映射到单个值
        
        # 4. 使用sigmoid函数将难度值压缩到0-1之间
        query_difficulty = torch.sigmoid(query_difficulty)
        
        # 5. 将难度值映射到代理数量范围
        agent_num_float = query_difficulty * self.max_agent
        # 例如：如果max_agent=6，难度0.5会映射到3个代理
        
        # 6. 将浮点数转换为整数，并限制在1到max_agent之间
        agent_num_int = torch.clamp(torch.round(agent_num_float), 1, self.max_agent).int()
        # clamp确保数量在[1, max_agent]范围内
        # round进行四舍五入
        # int()转换为整数类型
        
        # 7. 计算VAE的损失
        vae_loss = vae_loss_function(x_hat, queries, mu, log_var)
        # 包括重构损失和KL散度损失
        
        # 8. 返回结果
        return agent_num_int, agent_num_float, vae_loss
        # agent_num_int: 最终确定的代理数量（整数）
        # agent_num_float: 原始预测的代理数量（浮点数）
        # vae_loss: 用于模型训练的损失值

class RoleAllocation(torch.nn.Module):
    def __init__(self, input_dim:int=384, context_input_dim:int = 128, hidden_dim:int=64, temp=1.0, device=None):
        super().__init__()
        # 设置设备（CPU或GPU）
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化角色嵌入，用于记录已选角色的信息。初始大小为1*hidden_dim
        self.init_role_embedding = torch.zeros([1, hidden_dim],device=self.device,requires_grad=True)
        
        # VAE编码器，用于编码角色信息
        self.role_encoder = VAE(input_dim, hidden_dim, hidden_dim)
        
        # 上下文编码器，将查询和已选角色信息编码为隐藏表示
        self.context_encoder = nn.Linear(context_input_dim + hidden_dim, hidden_dim)
        
        # 角色-上下文融合器，用于融合角色和上下文信息
        self.role_context_encoder = GFusion(d_model=hidden_dim)
        
        # 温度参数，用于控制softmax的平滑程度
        self.temp = temp
        
    def forward(self, roles_list:List[torch.Tensor], contexts:torch.Tensor, agent_num_int:torch.Tensor):
        """
        输入:
            roles_list: 角色列表，每个元素是一个张量，形状为 N_r*input_dim
            contexts: 查询的上下文信息，形状为 N_q*input_dim，包含collab和query的嵌入向量
            agent_num_int: 每个查询需要的代理数量，形状为 N_q*1
        """
        # 存储选中的角色索引
        selected_roles_idx = []
        
        # 存储选择的对数概率
        log_probs = torch.zeros([contexts.size(0),1], device=self.device)
        
        # 存储角色摘要
        summary_role_list = []

        # roles_list包含多个角色列表。对每个角色列表进行处理
        for i, roles in enumerate(roles_list):
            selected_roles_idx.append([])
            
            # 使用VAE编码角色列表
            role_hat, role_z, role_mu, role_log_var = self.role_encoder(roles)
            role_embedding = F.normalize(role_z, p=2, dim=1)#对隐藏层做L2归一化

            # 计算VAE损失
            if i == 0:
                vae_loss = vae_loss_function(role_hat, roles, role_mu, role_log_var)
            else:
                vae_loss = vae_loss_function(role_hat, roles, role_mu, role_log_var) + vae_loss
                
            # 初始化当前和历史角色嵌入，初始化为全0
            current_role_embedding = self.init_role_embedding
            history_role_embedding = self.init_role_embedding

            # agent_num_int[i]表示当前任务需要的代理数量
            # 为每个代理选择角色
            for j in range(agent_num_int[i]):
                # 更新历史角色嵌入
                history_role_embedding = history_role_embedding + current_role_embedding
                # 层归一化，防止梯度爆炸
                history_role_embedding = F.layer_norm(history_role_embedding, history_role_embedding.shape[1:])

                # 编码上下文信息，把history_role_embedding连接到contexts[i]后面
                contexts_embedding = self.context_encoder(torch.cat([contexts[i].unsqueeze(0), history_role_embedding], dim=1))
                contexts_embedding = F.normalize(contexts_embedding, p=2, dim=1)

                # 计算 上下文信息 和 角色列表 的相似度分数
                scores = torch.matmul(contexts_embedding, role_embedding.T)
                scores = torch.softmax(scores/self.temp, dim=1)
                
                # 基于累积概率选择一个角色
                scores_cumsum = torch.cumsum(scores, dim=1)
                random_num = torch.rand([scores.size(0),1], device=self.device)
                selected_index = (scores_cumsum > random_num).float().argmax(dim=1)
                
                # 更新对数概率。log_probs[i][0]是当前任务的对数概率
                log_probs[i][0] = log_probs[i][0] + torch.log(scores[torch.arange(scores.size(0)), selected_index]).unsqueeze(1)

                # current_role_embedding是当前任务的角色嵌入
                current_role_embedding = role_embedding[selected_index]
                # 把选中的角色索引添加到selected_roles_idx
                selected_roles_idx[-1].append(selected_index)
                
            # 存储当前任务的角色摘要
            summary_role_list.append(history_role_embedding)
            
        # 合并所有任务的角色摘要
        summary_role = torch.cat(summary_role_list, dim=0)
        
        # 返回结果
        return selected_roles_idx, log_probs, summary_role, vae_loss/len(roles_list)
        # selected_roles_idx: 每个查询选中的角色索引列表
        # log_probs: 选择的对数概率
        # summary_role: 所有查询的角色摘要
        # vae_loss: 平均VAE损失

# 选择基座模型
class LLMRouter(torch.nn.Module):
    def __init__(self, input_dim:int=384, context_input_dim:int = 192, hidden_dim:int=64, temp:float=1.0, max_agent:int=6, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 用VAE编码基座模型
        self.llm_encoder = VAE(input_dim, hidden_dim, hidden_dim)
        # 用线性层编码上下文
        self.context_encoder = nn.Linear(context_input_dim, hidden_dim) # N_q*context_input_dim -> N_q*hidden_dim
        self.llm_context_encoder = GFusion(d_model=hidden_dim) # N_l*hidden_dim, N_q*hidden_dim -> N_l*hidden_dim
        self.temp = temp
        self.max_agent = max_agent

    def forward(self, llms:torch.Tensor, contexts:torch.Tensor, agent_num_int:torch.Tensor, agent_num_float:torch.Tensor):
        """
        输入参数:
            llms: N_l*input_dim tensor, N_l是LLM的数量，input_dim是每个LLM的维度
            contexts: N_q*input_dim tensor, N_q是查询的数量，input_dim是每个查询的维度
            agent_num_int: 每个查询需要的代理数量（整数形式）
            agent_num_float: 每个查询需要的代理数量（浮点数形式）
        """
        # 使用VAE编码LLM信息
        llm_hat, llm_z, llm_mu, llm_log_var = self.llm_encoder(llms) # N_l*hidden_dim
        # 对LLM的潜在表示进行L2归一化
        llm_embedding = F.normalize(llm_z, p=2, dim=1) # L2 normalization
        
        # 使用线性层编码上下文信息
        contexts_embedding = self.context_encoder(contexts) # N_q*hidden_dim
        # 对上下文的表示进行L2归一化
        contexts_embedding = F.normalize(contexts_embedding, p=2, dim=1) # L2 normalization
        
        # 计算VAE的损失（重构损失 + KL散度）
        vae_loss = vae_loss_function(llm_hat, llms, llm_mu, llm_log_var)

        # 计算上下文和LLM之间的相似度分数
        scores = torch.matmul(contexts_embedding, llm_embedding.T) # N_q*N_l
        # 使用温度参数进行softmax归一化，得到概率分布
        scores = torch.softmax(scores/self.temp, dim=1) # N_q*N_l
        # 计算累积概率分布
        scores_cumsum = torch.cumsum(scores, dim=1)
        
        # 初始化选择矩阵，用于记录每个查询选择的LLM
        selected_llm = torch.zeros([contexts.size(0), llms.size(0)], device=self.device) # N_q*N_l
        # 初始化选择的LLM索引列表
        selected_llm_index:List[List[int]] = [[] for i in range(contexts.size(0))] # 每个查询选择的LLM索引列表
        
        # 为每个代理位置选择LLM。注意这里并行处理了多个任务，所以要通过掩码确定哪些任务需要第i个代理
        for i in range(1, self.max_agent+1):
            # 创建掩码，标识哪些查询需要第i个代理
            agent_num_mask = (agent_num_int >= i).squeeze(1).float() # N_q
            # 生成随机数用于累积概率采样
            random_num = torch.rand_like(agent_num_float, device=self.device) # N_q*1
            # 基于累积概率选择LLM索引
            selected_index = (scores_cumsum > random_num).float().argmax(dim=1) # N_q
            # 更新选择矩阵
            selected_llm[torch.arange(selected_llm.size(0)), selected_index] += agent_num_mask # N_q*N_l

            # 将选中的LLM索引添加到结果列表中
            for j in range(contexts.size(0)):
                if agent_num_mask[j] > 0:#如果这个任务需要第i个代理
                    selected_llm_index[j].append(int(selected_index[j].item()))
        
        # 计算对数概率
        # 使用gamma函数计算组合数的对数
        # gammaln(agent_num_float + 1) - gammaln(selected_llm + 1).sum(dim=1) 计算组合数的对数
        # (selected_llm * torch.log(scores)).sum(dim=1) 计算选择概率的对数
        log_probs = gammaln(agent_num_float + 1) - gammaln(selected_llm + 1).sum(dim=1).unsqueeze(1) + (selected_llm * torch.log(scores)).sum(dim=1).unsqueeze(1) # N_q*1
        
        return selected_llm_index, log_probs, vae_loss
