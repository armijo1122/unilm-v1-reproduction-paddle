import paddle

def gather(x,dim,index):
    index_shape=index.shape
    index_flatten=index.flatten()
    if dim<0:
        dim=len(x.shape)+dim
    nd_index=[]
    for k in range(len(x.shape)):
        if k==dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape=[1]*len(x.shape)
            reshape_shape[k]=x.shape[k]
            dim_index=paddle.expand( paddle.reshape(paddle.arange(x.shape[k],dtype=index.dtype), reshape_shape), index_shape).flatten()
            nd_index.append(dim_index)

    ind2 = paddle.transpose(paddle.stack(nd_index),[1, 0])
    # ind2 = paddle.stack(nd_index).transpose([1, 0])
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out

def scatter_add_(paddle_out, dim, index, updates ):
    # assert  dim==0, "scatter_add_, no support dim>0"
    assert dim <=1, "no support dim>=1"
    if dim == 0:
        if "64" in str(updates.dtype):
            updates=updates.astype("float32")

        if "64" in str(paddle_out.dtype):
            paddle_out=paddle_out.astype("float32")
        if len(index.shape)==1:
            paddle.scatter_(paddle_out, index , updates.astype("float32"), overwrite=True)
        else:
            for ii in range(index.shape[1]):
                paddle.scatter_(paddle_out,index[:,ii],updates.astype("float32"),overwrite=True)
    if dim == 1:
        if "64" in str(updates.dtype):
            updates=updates.astype("float32")

        if "64" in str(paddle_out.dtype):
            paddle_out=paddle_out.astype("float32")
        for ii in range(index.shape[0]):
            paddle.scatter_(paddle_out,index[ii,:],updates.astype("float32"), overwrite=True)
            paddle_out = paddle.transpose([0,1])
    return paddle_out

def masked_fill_(masked, mask,value):
        mask=paddle.expand_as(mask,masked)
        new_values=paddle.where(mask,masked,paddle.ones(masked.shape)*value)
        paddle.assign(new_values,masked)

def clip_grad_value_(parameters, clip_value):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        paddle.clip(p.grad, min=-clip_value, max=clip_value)