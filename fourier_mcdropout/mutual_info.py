import torch
import numpy as np
def _eval_samples_entropy(mu_samples, std_samples):

    var_samples = std_samples ** 2

    mu_hat = mu_samples.mean(0)
    var_hat = var_samples.mean(0)

    K = mu_samples.shape[0]
    A = (mu_samples - mu_hat).T / np.sqrt(K)

    IK = torch.eye(K).to(mu_samples.device)

    logdet_term1 = torch.log(var_hat).sum()
    logdet_term2 = torch.logdet(A.T @ torch.diag(1 / var_hat) @ A + IK)
    entropy = logdet_term1 + logdet_term2

    return entropy

def _eval_sf_mutual_info(pred_mu_samples, pred_std_samples, Fmu_samples, Fstd_samples):

    Fmu_samples_flat = Fmu_samples.flatten(2,-1)
    Fstd_samples_flat = Fstd_samples.flatten(2,-1)
    assert Fmu_samples_flat.shape[0] == Fstd_samples_flat.shape[0]
    nF = Fmu_samples_flat.shape[0]

    pred_info = []

    for i in range(pred_mu_samples.shape[0]):
        preds_mu=pred_mu_samples[i,...]
        preds_std=pred_std_samples[i,...]
        if pred_mu_samples.shape[1] != Fmu_samples.shape[1]:

            target_size = (87,87)
            # 使用 interpolate 函数进行插
            preds_mu=preds_mu.unsqueeze(0).unsqueeze(0)
            preds_std=preds_std.unsqueeze(0).unsqueeze(0)

            hf_preds_std =  F.interpolate(preds_mu, size=target_size, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
            hf_preds_mu=F.interpolate(preds_std, size=target_size, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

            hf_preds_mu = hf_preds_mu.squeeze(0)
            hf_preds_std = hf_preds_std.squeeze(0)
            preds_mu = hf_preds_mu.squeeze(0)
            preds_std = hf_preds_std.squeeze(0)

        preds_mu =preds_mu.flatten(1, -1)
        preds_std = preds_std.flatten(1, -1)
        # print("i sample", preds_mu.shape,preds_std.shape)
        Hx = _eval_samples_entropy(preds_mu, preds_std)
        info_list = []

        for s in range(nF):
            Fmu_s = Fmu_samples_flat[s,:]
            Fstd_s = Fstd_samples_flat[s,:]
            # print(Fmu_s.shape, Fstd_s.shape)
            HF = _eval_samples_entropy(Fmu_s, Fstd_s)

            HxF = _eval_samples_entropy(
                torch.hstack([preds_mu, Fmu_s]),
                torch.hstack([preds_std, Fstd_s]),
            )


            info_s = Hx + HF - HxF
            info_list.append(info_s)


        info = sum(info_list)

        # print(info)

        pred_info.append(info.item())
    #

    pred_info = np.array(pred_info)

    # print(pred_info)

    return pred_info
def max_cost_indices_within_budget(budget,data,index_id):
    total_cost = 0
    max_cost = 0
    selected_indices = []
    # print(index_id.shape,data.shape)
    for index in index_id: #[2,0,1]
        cost = 1 if data[index,1] == 0 else 4

        if total_cost + cost <= budget:
            total_cost += cost
            max_cost = max(max_cost, cost)
            selected_indices.append(index)


    return selected_indices


def mf_mutual_info(mf_predicts,acq_samples):

    batch_budget=80
    mu_samples,std_samples=mf_predicts
    Fidx = np.random.choice(
        acq_samples,
        size=200,
        replace=False
    )
    Fmu_low= mu_samples[-2][Fidx]
    Fstd_low = std_samples[-2][Fidx]
    Fmu_high = mu_samples[-1][Fidx]
    Fstd_high = std_samples[-1][Fidx]
    # print(mu_samples[-1][Fidx].shape,mu_samples[-2][Fidx].shape)
    mf_hvals = []
    mf_low=[]
    mf_high=[]
    # for pred_mu_samples, pred_std_samples in zip(mu_samples, std_samples):
    #     # print("weired",pred_mu_samples.shape,
    #     #     Fmu_samples.shape)
    #     sf_info = _eval_sf_mutual_info(pred_mu_samples, pred_std_samples, Fmu_low, Fstd_low)
    #     mf_hvals.append(sf_info)
    # for pred_mu_samples, pred_std_samples in zip(mu_samples[-2], std_samples[-2]):
    #     # print("weired",pred_mu_samples.shape,
    #     #     Fmu_samples.shape)
    #     print(pred_mu_samples.shape,pred_std_samples.shape)
    sf_info_1= _eval_sf_mutual_info(mu_samples[-2], std_samples[-2], Fmu_low, Fstd_low)
    mf_low.append(sf_info_1)
    mf_low=np.array(mf_low)
    sf_info_2 = _eval_sf_mutual_info(mu_samples[-1], std_samples[-1], Fmu_high, Fstd_high)
    mf_high.append(sf_info_2)
    mf_high=np.array(mf_high)
    mf_hvals.append(mf_low)
    mf_hvals.append(mf_high)
    # print("sf",sf_info_1[:600,...]/sf_info_2[:600,...],np.mean(sf_info_1[:600,...]/sf_info_2[:600,...]))
    # print(sf_info) # 6512 6453 6428 6544 6422\
    # print("mf_vals",len(mf_hvals))
    fids_costs=[1,1]
    mf_query_mat = []
    for fidx, hvals in enumerate(mf_hvals):
        # print(f"fid {fidx},hvals {len(hvals),type(hvals)}")
        cost_i = fids_costs[fidx]
        nhvals = hvals/cost_i
        vec_fid_idx = np.ones_like(hvals) * fidx
        vec_data_idx = np.arange(nhvals.size)
        sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T # nhvals? vec_fid_id? vec_data_idx?
        # print("nhvals",nhvals,vec_fid_idx,vec_data_idx)# values fidelity index
        mf_query_mat.append(sf_query_mat)
    #
    # print('mf_query', len(mf_query_mat),mf_query_mat[0].shape) # 200 3
    mf_query_mat = np.vstack(mf_query_mat)
    # print('mf_query', type(mf_query_mat),mf_query_mat.shape) # 290 3

    if np.isnan(mf_query_mat).any():
        raise Exception('Error: nan found in acquisition')

    ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
    # sorted_indices = np.argsort(mf_query_mat[:, 0])[::-1]

    # 根据排序后的索引重新排列数组的行
    # mf_query_mat =mf_query_mat[sorted_indices]
    # print("order",ordered_idx)
    # print('ordered', mf_query_mat[0],mf_query_mat[1],mf_query_mat[2]) # 290 3
    argmax_idx = max_cost_indices_within_budget(batch_budget,mf_query_mat,ordered_idx)

    queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
    queries_input = mf_query_mat[argmax_idx, 2].astype(int)
    return queries_fid, queries_input
def _eval_sf_predvar(pred_samples):
    pred_samples_flat = pred_samples.flatten(2,-1)
    # cprint('r', pred_samples_flat.shape)
    pred_var = pred_samples_flat.std(1).mean(1).square()
    # cprint('b', pred_var.shape)
    return pred_var.data.cpu().numpy()
def mf_predvar_info(mf_predicts,acq_samples):

    batch_budget=80
    mu_samples,std_samples=mf_predicts
    mf_predvars=[]
    for sf_mu_samples in mu_samples:
        sf_pred_var=_eval_sf_predvar(sf_mu_samples)
        mf_predvars.append(sf_pred_var)

    mf_query_mat=[]
    fids_costs=[1,1]
    for fidx, hvals in enumerate(mf_predvars):
        cost_i = fids_costs[fidx]
        nhvals = hvals/cost_i
        vec_fid_idx = np.ones_like(hvals) * fidx
        vec_data_idx = np.arange(nhvals.size)
        sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T # nhvals? vec_fid_id? vec_data_idx?
        # print("nhvals",nhvals,vec_fid_idx,vec_data_idx)# values fidelity index
        mf_query_mat.append(sf_query_mat)
    #
    # print('mf_query', len(mf_query_mat),mf_query_mat[0].shape) # 200 3
    mf_query_mat = np.vstack(mf_query_mat)
    # print('mf_query', type(mf_query_mat),mf_query_mat.shape) # 290 3

    if np.isnan(mf_query_mat).any():
        raise Exception('Error: nan found in acquisition')

    ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
    # sorted_indices = np.argsort(mf_query_mat[:, 0])[::-1]

    # 根据排序后的索引重新排列数组的行
    # mf_query_mat =mf_query_mat[sorted_indices]
    # print("order",ordered_idx)
    # print('ordered', mf_query_mat[0],mf_query_mat[1],mf_query_mat[2]) # 290 3
    argmax_idx = max_cost_indices_within_budget(batch_budget,mf_query_mat,ordered_idx)

    queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
    queries_input = mf_query_mat[argmax_idx, 2].astype(int)
    return queries_fid, queries_input
import torch.nn.functional as F
# def inte(x,target):
#     layer=x.shape[-1]
#     inter=[]
#     for i in range(layer):
#         hf_preds_std = F.interpolate(hf_preds_std, size=target_size, mode='bilinear', align_corners=False)
#         hf_preds_mu = F.interpolate(hf_preds_mu, size=target_size, mode='bilinear', align_corners=False)
def _eval_sf_self_mutual_info(
        pred_mu_samples,
        pred_std_samples,
        hf_pred_mu_samples,
        hf_pred_std_samples
):
    pred_info = []
    # print("***",pred_mu_samples.shape,hf_pred_mu_samples.shape) # 600 580
    # [2000 55 55] [2000,87,87]
    for i in range(min(pred_mu_samples.shape[0],pred_mu_samples.shape[0])):
        preds_mu=pred_mu_samples[i, ...]
        preds_std=pred_std_samples[i, ...]
        hf_preds_mu=hf_pred_mu_samples[i, ...]
        hf_preds_std=hf_pred_std_samples[i, ...]
        # if preds_mu.shape[1] != hf_preds_mu.shape[1]:
        #     target_size = (55,55)
        #     # 使用 interpolate 函数进行插值
        #     hf_preds_std =  F.interpolate(preds_std.permute(2, 0, 1).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        #     hf_preds_mu=F.interpolate(preds_mu.permute(2, 0, 1).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        #     # print(hf_preds_mu.shape,hf_preds_std.shape)
        #     hf_preds_mu = hf_preds_mu.squeeze(0)
        #     hf_preds_std = hf_preds_std.squeeze(0)
        #     hf_preds_mu = hf_preds_mu.squeeze(0)
        #     hf_preds_std = hf_preds_std.squeeze(0)
        if preds_mu.shape[1] != hf_preds_mu.shape[1]:
            target_size = (55,55)
            # 使用 interpolate 函数进行插
            preds_mu=preds_mu.unsqueeze(0).unsqueeze(0)
            preds_std=preds_std.unsqueeze(0).unsqueeze(0)

            hf_preds_std =  F.interpolate(preds_std, size=target_size, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
            hf_preds_mu=F.interpolate(preds_mu, size=target_size, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

            hf_preds_mu = hf_preds_mu.squeeze(0)
            hf_preds_std = hf_preds_std.squeeze(0)
            hf_preds_mu = hf_preds_mu.squeeze(0)
            hf_preds_std = hf_preds_std.squeeze(0)
            preds_mu=preds_mu.squeeze(0).squeeze(0)
            preds_std=preds_std.squeeze(0).squeeze(0)
        preds_mu = preds_mu.flatten(1, -1)
        preds_std = preds_std.flatten(1, -1)
        hf_preds_mu = hf_preds_mu.flatten(1,-1)
        hf_preds_std = hf_preds_std.flatten(1,-1)
        Hx = _eval_samples_entropy(preds_mu, preds_std)
        HF = _eval_samples_entropy(hf_preds_mu, hf_preds_std)
        HxF = _eval_samples_entropy(
            torch.hstack([preds_mu, hf_preds_mu]),
            torch.hstack([preds_std, hf_preds_std]),
        )

        info = Hx+HF-HxF
        pred_info.append(info.item())
    #

    pred_info = np.array(pred_info)
    return pred_info

def mf_self_mutual_info(mf_predicts):

    batch_budget=80
    mu_samples,std_samples=mf_predicts
    mf_hvals=[]
    for pred_mu_samples, pred_std_samples in zip(mu_samples,std_samples):
        hf_pred_mu_samples = mu_samples[-1]
        hf_pred_std_samples = std_samples[-1]
        # 600 580
        sf_info = _eval_sf_self_mutual_info(
            pred_mu_samples,
            pred_std_samples,
            hf_pred_mu_samples,
            hf_pred_std_samples
        )
        mf_hvals.append(sf_info)


    mf_query_mat = []
    fids_costs=[1,1]

    for fidx, hvals in enumerate(mf_hvals):
        cost_i = fids_costs[fidx]
        nhvals = hvals / cost_i
        vec_fid_idx = np.ones_like(hvals) * fidx
        vec_data_idx = np.arange(nhvals.size)
        sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
        mf_query_mat.append(sf_query_mat)
    mf_query_mat = np.vstack(mf_query_mat)
    if np.isnan(mf_query_mat).any():
        raise Exception('Error: nan found in acquisition')

    ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
    argmax_idx = max_cost_indices_within_budget(batch_budget,mf_query_mat,ordered_idx)

    queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
    queries_input = mf_query_mat[argmax_idx, 2].astype(int)
    return queries_fid, queries_input


def mf_self_mutual_info(mf_predicts):

    batch_budget=80
    mu_samples,std_samples=mf_predicts
    mf_hvals=[]
    for pred_mu_samples, pred_std_samples in zip(mu_samples,std_samples):
        hf_pred_mu_samples = mu_samples[-1]
        hf_pred_std_samples = std_samples[-1]
        # 600 580
        sf_info = _eval_sf_self_mutual_info(
            pred_mu_samples,
            pred_std_samples,
            hf_pred_mu_samples,
            hf_pred_std_samples
        )
        mf_hvals.append(sf_info)


    mf_query_mat = []
    fids_costs=[1,1]

    for fidx, hvals in enumerate(mf_hvals):
        cost_i = fids_costs[fidx]
        nhvals = hvals / cost_i
        vec_fid_idx = np.ones_like(hvals) * fidx
        vec_data_idx = np.arange(nhvals.size)
        sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
        mf_query_mat.append(sf_query_mat)
    mf_query_mat = np.vstack(mf_query_mat)
    if np.isnan(mf_query_mat).any():
        raise Exception('Error: nan found in acquisition')

    ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
    argmax_idx = max_cost_indices_within_budget(batch_budget,mf_query_mat,ordered_idx)

    queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
    queries_input = mf_query_mat[argmax_idx, 2].astype(int)
    return queries_fid, queries_input

def _eval_sf_full_entropy(pred_mu_samples, pred_std_samples):

    pred_info = []
    for i in range(pred_mu_samples.shape[0]):

        preds_mu = pred_mu_samples[i, ...].flatten(1, -1)
        preds_std = pred_std_samples[i, ...].flatten(1, -1)

        # cprint('r', preds_mu.shape)
        # cprint('b', preds_std.shape)

        Hx = _eval_samples_entropy(preds_mu, preds_std)

        pred_info.append(Hx.item())
    #

    pred_info = np.array(pred_info)

    return pred_info

def mf_self_mutual_info_3d(mf_predicts,batch_budget,rate):


    mu_samples,std_samples=mf_predicts
    mf_hvals=[]
    fidx=np.random.choice(
    mu_samples[-1].shape[0],
        size=50,
        replace=False
    )

    index=0
    for pred_mu_samples, pred_std_samples in zip(mu_samples,std_samples):
        # hf_pred_mu_samples=mu_samples[-1]
        # hf_pred_std_samples=std_samples[-1]
        info_buff=[]
        if index==0:
            flag=-2
            index=1
        else:
            flag=-1
        fmu_samples = mu_samples[flag][fidx]
        fstd_samples = std_samples[flag][fidx]
        for t in range(pred_mu_samples.shape[-1]):# 500 55 55 2 1

            sf_info_t = _eval_sf_mutual_info(
                pred_mu_samples[..., t].to("cuda"),
                pred_std_samples[..., t].to("cuda"),
                fmu_samples[..., t].to("cuda"),
                fstd_samples[..., t].to("cuda")
            )
            # sf_info_t = _eval_sf_full_entropy(
            #     pred_mu_samples[..., t],
            #     pred_std_samples[..., t]
            # )
            # info_buff.append(sf_info_t)
            info_buff.append(sf_info_t)
        sf_info=sum(info_buff)
        mf_hvals.append(sf_info)

    mf_query_mat = []
    fids_costs=[1,rate]

    for fidx, hvals in enumerate(mf_hvals):
        cost_i = fids_costs[fidx]
        nhvals = hvals / cost_i
        vec_fid_idx = np.ones_like(hvals) * fidx
        vec_data_idx = np.arange(nhvals.size)
        sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
        sorted_sf_query_mat = sf_query_mat[np.argsort(-sf_query_mat[:, 0])]
        # 打印前20个第一维度的数值
        print(sorted_sf_query_mat[:20, 0])
        mf_query_mat.append(sf_query_mat)
        print("-----")

    mf_query_mat = np.vstack(mf_query_mat)
    if np.isnan(mf_query_mat).any():
        raise Exception('Error: nan found in acquisition')
   # 4000 3
    ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
    rate=rate*0.98
    # mf_query  [[5 2 3],[3,2,5],[9,7,8]
    # argsort [1,0,2]
    # flip [2,0,1]

    argmax_idx = max_cost_indices_within_budget(batch_budget,mf_query_mat,ordered_idx)

    queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
    queries_input = mf_query_mat[argmax_idx, 2].astype(int)
    return queries_fid, queries_input,rate


def mf_self_mutual_info_4d(mf_predicts,batch_budget,rate):


    mu_samples,std_samples=mf_predicts
    mu_samples[0]=mu_samples[0].reshape(mu_samples[0].shape[0],mu_samples[0].shape[1],mu_samples[0].shape[2],-1)
    std_samples[0]=std_samples[0].reshape(std_samples[0].shape[0],std_samples[0].shape[1],std_samples[0].shape[2],-1)
    mu_samples[1] = mu_samples[1].reshape(mu_samples[1].shape[0], mu_samples[1].shape[1], mu_samples[1].shape[2], -1)
    std_samples[1] = std_samples[1].reshape(std_samples[1].shape[0], std_samples[1].shape[1], std_samples[1].shape[2],
                                            -1)
    mf_hvals=[]
    fidx=np.random.choice(
    mu_samples[-1].shape[0],
        size=100,
        replace=False
    )

    index=0
    for pred_mu_samples, pred_std_samples in zip(mu_samples,std_samples):
        # hf_pred_mu_samples=mu_samples[-1]
        # hf_pred_std_samples=std_samples[-1]
        info_buff=[]
        if index==0:
            flag=-2
            index=1
        else:
            flag=-1
        fmu_samples = mu_samples[flag][fidx]
        fstd_samples = std_samples[flag][fidx]
        for t in range(pred_mu_samples.shape[-1]):# 500 55 55 2 1

            sf_info_t = _eval_sf_mutual_info(
                pred_mu_samples[..., t].to("cuda"),
                pred_std_samples[..., t].to("cuda"),
                fmu_samples[..., t].to("cuda"),
                fstd_samples[..., t].to("cuda")
            )

            info_buff.append(sf_info_t)
        sf_info=sum(info_buff)
        mf_hvals.append(sf_info)

    mf_query_mat = []
    fids_costs=[1,rate]

    for fidx, hvals in enumerate(mf_hvals):
        cost_i = fids_costs[fidx]
        nhvals = hvals / cost_i
        vec_fid_idx = np.ones_like(hvals) * fidx
        vec_data_idx = np.arange(nhvals.size)
        sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
        sorted_sf_query_mat = sf_query_mat[np.argsort(-sf_query_mat[:, 0])]
        # 打印前20个第一维度的数值
        mf_query_mat.append(sf_query_mat)


    mf_query_mat = np.vstack(mf_query_mat)
    if np.isnan(mf_query_mat).any():
        raise Exception('Error: nan found in acquisition')
   # 4000 3
    ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
    rate=rate*1.0005
    # mf_query  [[5 2 3],[3,2,5],[9,7,8]
    # argsort [1,0,2]
    # flip [2,0,1]

    argmax_idx = max_cost_indices_within_budget(batch_budget,mf_query_mat,ordered_idx)

    queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
    queries_input = mf_query_mat[argmax_idx, 2].astype(int)
    return queries_fid, queries_input,rate


def mf_self_mutual_info_3d(mf_predicts,batch_budget,rate):


    mu_samples,std_samples=mf_predicts
    mf_hvals=[]
    fidx=np.random.choice(
    mu_samples[-1].shape[0],
        size=100,
        replace=False
    )

    index=0
    for pred_mu_samples, pred_std_samples in zip(mu_samples,std_samples):
        # hf_pred_mu_samples=mu_samples[-1]
        # hf_pred_std_samples=std_samples[-1]
        info_buff=[]
        if index==0:
            flag=-2
            index=1
        else:
            flag=-1
        fmu_samples = mu_samples[flag][fidx]
        fstd_samples = std_samples[flag][fidx]
        for t in range(pred_mu_samples.shape[-1]):# 500 55 55 2 1

            sf_info_t = _eval_sf_mutual_info(
                pred_mu_samples[..., t].to("cuda"),
                pred_std_samples[..., t].to("cuda"),
                fmu_samples[..., t].to("cuda"),
                fstd_samples[..., t].to("cuda")
            )

            info_buff.append(sf_info_t)
        sf_info=sum(info_buff)
        mf_hvals.append(sf_info)

    mf_query_mat = []
    fids_costs=[1,rate]

    for fidx, hvals in enumerate(mf_hvals):
        cost_i = fids_costs[fidx]
        nhvals = hvals / cost_i
        vec_fid_idx = np.ones_like(hvals) * fidx
        vec_data_idx = np.arange(nhvals.size)
        sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
        sorted_sf_query_mat = sf_query_mat[np.argsort(-sf_query_mat[:, 0])]
        # 打印前20个第一维度的数值
        mf_query_mat.append(sf_query_mat)


    mf_query_mat = np.vstack(mf_query_mat)
    if np.isnan(mf_query_mat).any():
        raise Exception('Error: nan found in acquisition')
   # 4000 3
    ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
    # rate=rate*1.005
    # mf_query  [[5 2 3],[3,2,5],[9,7,8]
    # argsort [1,0,2]
    # flip [2,0,1]

    argmax_idx = max_cost_indices_within_budget(batch_budget,mf_query_mat,ordered_idx)

    queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
    queries_input = mf_query_mat[argmax_idx, 2].astype(int)
    return queries_fid, queries_input,rate

