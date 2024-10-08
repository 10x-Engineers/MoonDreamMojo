from max.engine import InputSpec, InferenceSession
from python import Python, PythonObject
from utils.index import Index
from time import now
from max.graph import Graph, TensorType, Type, ops, Symbol
from max import engine
from max.tensor import Tensor, TensorShape
from max.engine import Model
from algorithm import sum
from utils.numerics import inf
from algorithm import parallelize
from memory import memcpy, memcmp, memset_zero
from max.graph.checkpoint import save, TensorDict, load


alias num_attention_heads = 32
alias hidden_size = 2048
alias head_dim = hidden_size // num_attention_heads
alias min_len = 700
alias pi_sqrt = 0.7978845608028654
alias batch_size = 1


fn numpy_to_tensor(numpy_array: PythonObject) raises -> Tensor[DType.float32]:
    var tensor_shape = numpy_array.shape
    var tensor_rank = len(numpy_array.shape)
    var shape_list: List[Int]  = List[Int]()
    for i in range(tensor_rank):
        shape_list.append(tensor_shape[i].__int__())

    var tensor = Tensor[DType.float32] (shape_list)

    memcpy(tensor.unsafe_ptr(), numpy_array.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float32](), 
           tensor.num_elements())
        
    return tensor


fn tensor_to_numpy(tensor: Tensor[DType.float32]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var tensor_shape = tensor.shape()
    var tensor_rank = tensor.rank()

    var python_list = Python.evaluate("list()")
    for i in range(tensor_rank):
        _ = python_list.append(tensor_shape[i])

    var numpy_array:PythonObject = np.zeros(python_list, dtype=np.float32)
    memcpy(numpy_array.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float32](), tensor.unsafe_ptr(), 
           tensor.num_elements())
    return numpy_array^

struct LayerNorm(CollectionElement):
    var w: Tensor[DType.float32]
    var b: Tensor[DType.float32]

    fn __init__(inout self, gema: Tensor[DType.float32], beta: Tensor[DType.float32]):
        self.w = gema
        self.b = beta

    fn __copyinit__(inout self, existing: Self):
        self.w = existing.w
        self.b = existing.b
    
    fn __moveinit__(inout self, owned existing: Self):
        self.w = existing.w^
        self.b = existing.b^
    
    fn forward(self, inputs_embeds: Tensor[DType.float32], mean: Model) raises -> Tensor[DType.float32]:
        results = mean.execute("input0", inputs_embeds)
        m = results.get[DType.float32]("output0")
        variance = Tensor[DType.float32] (m.shape())
        var epsilon:Float32 = 1e-5

        for b in range(inputs_embeds.shape()[0]):
            for i in range(inputs_embeds.shape()[1]):
                var sum_squared_diff:Float32 = 0.0
                for j in range(inputs_embeds.shape()[2]):
                    diff = inputs_embeds[Index(b, i, j)] - m[Index(b, i, 0)]
                    sum_squared_diff += diff ** 2

                variance[Index(b, i, 0)] = sum_squared_diff / inputs_embeds.shape()[2]

        x = Tensor[DType.float32] (inputs_embeds.shape())
        y = Tensor[DType.float32] (inputs_embeds.shape())
        for i in range(x.shape()[0]):
            for j in range(x.shape()[1]):
                for k in range(x.shape()[2]):
                    x[Index(i,j,k)] = inputs_embeds[Index(i,j,k)] - m[j]

        y = variance + epsilon

        var exponent = Float32(0.5)
        for i in range(y.shape()[0]):
            for j in range(y.shape()[1]):
                for k in range(y.shape()[2]):
                    var base = Float32(y[Index(i,j,k)])
                    y[Index(i,j,k)] =  pow(base, exponent)


        z = Tensor[DType.float32] (inputs_embeds.shape())
        for i in range(z.shape()[0]):
            for j in range(z.shape()[1]):
                for k in range(z.shape()[2]):
                    z[Index(i,j,k)] = y[Index(0,j,0)]
        
        normalized_states = x/z

        out = Tensor[DType.float32] (normalized_states.shape())

        for i in range(out.shape()[0]):
            for j in range(out.shape()[1]):
                for k in range(out.shape()[2]):
                    out[Index(i,j,k)] = normalized_states[Index(i,j,k)] * self.w[k] + self.b[k]
        return out

struct Linear(CollectionElement):
    var w: Tensor[DType.float32]
    var b: Tensor[DType.float32]

    fn __init__(inout self, w: Tensor[DType.float32], b: Tensor[DType.float32]):
        self.w = w
        self.b = b

    fn __copyinit__(inout self, existing: Self):
        self.w = existing.w
        self.b = existing.b
    
    fn __moveinit__(inout self, owned existing: Self):
        self.w = existing.w^
        self.b = existing.b^
    
    fn forward(self, inputs_mat: Tensor[DType.float32], transpose: Model, multiplication: Model, addition: Model) 
               raises -> Tensor[DType.float32]:
        var results = transpose.execute("input0", self.w)
        var W_T = results.get[DType.float32]("output0")
        results = multiplication.execute("input0", inputs_mat, "input1", W_T)
        var out = results.get[DType.float32]("output0")
        results = addition.execute("input0", out, "input1", self.b)
        lin_out = results.get[DType.float32]("output0")
        return lin_out


struct QKVstates(CollectionElement):
    var bsz: Int
    var q_len: Int
    var qkv: Tensor[DType.float32]

    fn __init__(inout self, bsz: Int, q_len:Int, qkv: Tensor[DType.float32]):
        self.bsz = bsz
        self.q_len = q_len
        self.qkv = qkv

    fn __copyinit__(inout self, existing: Self):
        self.bsz = existing.bsz
        self.q_len = existing.q_len
        self.qkv = existing.qkv
    
    fn __moveinit__(inout self, owned existing: Self):
        self.bsz = existing.bsz
        self.q_len = existing.q_len
        self.qkv = existing.qkv^
    
    fn chunk_qkv(self, qkv: Tensor[DType.float32], inout q: Tensor[DType.float32], inout start_qkv: Int):
        var start_q = 0
        alias load_sizes = 2048
        var q_num_elements = q.num_elements()
        for i in range(0, q_num_elements, load_sizes):
            q.store(start_q, qkv.load[width=load_sizes](start_qkv))
            start_q += load_sizes
            start_qkv += load_sizes*3

    fn forward(self, transpose_12: Model) raises -> List[Tensor[DType.float32]]:
        var t1:TensorShape = (self.qkv.shape()[0], self.qkv.shape()[1], int(self.qkv.shape()[2]/3))
        var query_states = Tensor[DType.float32] (t1)
        var key_states = Tensor[DType.float32] (t1)
        var value_states = Tensor[DType.float32] (t1)

        var start_qkv_q = 0
        var start_qkv_k = 2048
        var start_qkv_v = 4096
        self.chunk_qkv(self.qkv, query_states, start_qkv_q)
        self.chunk_qkv(self.qkv, key_states, start_qkv_k)
        self.chunk_qkv(self.qkv, value_states, start_qkv_v)

        reshaped_q = query_states.reshape((self.bsz, self.q_len, num_attention_heads, head_dim))
        results = transpose_12.execute("input0", reshaped_q)
        query_states = results.get[DType.float32]("output0")

        reshaped_k = key_states.reshape((self.bsz, self.q_len, num_attention_heads, head_dim))
        results = transpose_12.execute("input0", reshaped_k)
        key_states = results.get[DType.float32]("output0")

        reshaped_v = value_states.reshape((self.bsz, self.q_len, num_attention_heads, head_dim))
        results = transpose_12.execute("input0", reshaped_v)
        value_states = results.get[DType.float32]("output0")

        out = List[Tensor[DType.float32]] ()
        out.append(query_states)
        out.append(key_states)
        out.append(value_states)

        return out


struct RotPass(CollectionElement):
    var query_states: Tensor[DType.float32]
    var key_states: Tensor[DType.float32]

    fn __init__(inout self, query_states: Tensor[DType.float32], key_states: Tensor[DType.float32]):
        self.query_states = query_states
        self.key_states = key_states

    fn __copyinit__(inout self, existing: Self):
        self.query_states = existing.query_states
        self.key_states = existing.key_states
    
    fn __moveinit__(inout self, owned existing: Self):
        self.query_states = existing.query_states^
        self.key_states = existing.key_states^
    
    fn forward(self) raises -> List[Tensor[DType.float32]]:
        
        query_rot = Tensor[DType.float32] (self.query_states.shape()[0], self.query_states.shape()[1], 
                                           self.query_states.shape()[2], num_attention_heads)
        for i in range(query_rot.shape()[0]):
            for j in range(query_rot.shape()[1]):
                for k in range(query_rot.shape()[2]):
                    for l in range(query_rot.shape()[3]):
                        query_rot[Index(i,j,k,l)] = self.query_states[Index(i,j,k,l)]


        query_pass = Tensor[DType.float32] (self.query_states.shape()[0], self.query_states.shape()[1], 
                                            self.query_states.shape()[2], num_attention_heads)
        for i in range(query_pass.shape()[0]):
            for j in range(query_pass.shape()[1]):
                for k in range(query_pass.shape()[2]):
                    for l in range(query_pass.shape()[3]):
                        query_pass[Index(i,j,k,l)] = self.query_states[Index(i,j,k,l+num_attention_heads)]


        key_rot = Tensor[DType.float32] (self.key_states.shape()[0], self.key_states.shape()[1], 
                                         self.key_states.shape()[2], num_attention_heads)
        for i in range(key_rot.shape()[0]):
            for j in range(key_rot.shape()[1]):
                for k in range(key_rot.shape()[2]):
                    for l in range(key_rot.shape()[3]):
                        key_rot[Index(i,j,k,l)] = self.key_states[Index(i,j,k,l)]


        key_pass = Tensor[DType.float32] (self.key_states.shape()[0], self.key_states.shape()[1], 
                                          self.key_states.shape()[2], num_attention_heads)
        for i in range(key_pass.shape()[0]):
            for j in range(key_pass.shape()[1]):
                for k in range(key_pass.shape()[2]):
                    for l in range(key_pass.shape()[3]):
                        key_pass[Index(i,j,k,l)] = self.key_states[Index(i,j,k,l+num_attention_heads)]
        
        out = List[Tensor[DType.float32]] ()
        out.append(query_rot)
        out.append(query_pass)
        out.append(key_rot)
        out.append(key_pass)

        return out

struct RotPosEmb(CollectionElement):
    var cos: Tensor[DType.float32]
    var sin: Tensor[DType.float32]
    var pos_ids: Tensor[DType.float32]

    fn __init__(inout self, cos: Tensor[DType.float32], sin: Tensor[DType.float32], pos_ids: Tensor[DType.float32]):
        self.cos = cos
        self.sin = sin
        self.pos_ids = pos_ids

    fn __copyinit__(inout self, existing: Self):
        self.cos = existing.cos
        self.sin = existing.sin
        self.pos_ids = existing.pos_ids
    
    fn __moveinit__(inout self, owned existing: Self):
        self.cos = existing.cos^
        self.sin = existing.sin^
        self.pos_ids = existing.pos_ids^
    
    fn forward(self, q:Tensor[DType.float32], k:Tensor[DType.float32]) raises ->List[Tensor[DType.float32]]:
        new_cos = Tensor[DType.float32] (self.pos_ids.shape()[0], self.pos_ids.shape()[1], self.cos.shape()[1])
        new_sin = Tensor[DType.float32] (self.pos_ids.shape()[0], self.pos_ids.shape()[1], self.cos.shape()[1])
        for i in range(new_cos.shape()[0]):
            for j in range(new_cos.shape()[1]):
                pos_id = self.pos_ids[i, j]
                for k in range(new_cos.shape()[2]):
                    new_cos[Index(i,j,k)] = self.cos[Index(pos_id, k)]
                    new_sin[Index(i,j,k)] = self.sin[Index(pos_id, k)]

        
        new_cos = new_cos.reshape((1,self.pos_ids.shape()[0], self.pos_ids.shape()[1], self.cos.shape()[1]))
        new_sin = new_sin.reshape((1,self.pos_ids.shape()[0], self.pos_ids.shape()[1], self.cos.shape()[1]))

        rotate_half_q_x1 = Tensor[DType.float32] (q.shape()[0], q.shape()[1], q.shape()[2], int(q.shape()[3]//2))
        rotate_half_q_x2 = Tensor[DType.float32] (q.shape()[0], q.shape()[1], q.shape()[2], int(q.shape()[3]//2))

        rotate_half_k_x1 = Tensor[DType.float32] (k.shape()[0], k.shape()[1], k.shape()[2], int(k.shape()[3]//2))
        rotate_half_k_x2 = Tensor[DType.float32] (k.shape()[0], k.shape()[1], k.shape()[2], int(k.shape()[3]//2))

        for b in range(rotate_half_q_x1.shape()[0]):
            for h in range(rotate_half_q_x1.shape()[1]):
                for i in range(rotate_half_q_x1.shape()[2]):
                    for j in range(rotate_half_q_x1.shape()[3]):
                        # First half remains the same
                        rotate_half_q_x1[Index(b, h, i, j)] = q[Index(b, h, i, j)]
                        rotate_half_k_x1[Index(b, h, i, j)] = k[Index(b, h, i, j)]
                        # Second half is negated and swapped
                        rotate_half_q_x2[Index(b, h, i, j)] = -1 * q[Index(b, h, i, j + 16)]
                        rotate_half_k_x2[Index(b, h, i, j)] = -1 * k[Index(b, h, i, j + 16)]

        
        rotate_half_q_out = Tensor[DType.float32] (rotate_half_q_x1.shape()[0],rotate_half_q_x1.shape()[1],
                                                rotate_half_q_x1.shape()[2], rotate_half_q_x1.shape()[3] + rotate_half_q_x2.shape()[3])
        rotate_half_k_out = Tensor[DType.float32] (rotate_half_k_x1.shape()[0],rotate_half_k_x1.shape()[1],
                                                rotate_half_k_x1.shape()[2], rotate_half_k_x1.shape()[3] + rotate_half_k_x2.shape()[3])
        
        for b in range(rotate_half_q_x1.shape()[0]):
            for h in range(rotate_half_q_x1.shape()[1]):
                for i in range(rotate_half_q_x1.shape()[2]):
                    for j in range(rotate_half_q_x1.shape()[3]):
                        # First 16 elements: negated -x2
                        rotate_half_q_out[Index(b, h, i, j)] = rotate_half_q_x2[Index(b, h, i, j)]
                        rotate_half_k_out[Index(b, h, i, j)] = rotate_half_k_x2[Index(b, h, i, j)]
                        # Next 16 elements: x1
                        rotate_half_q_out[Index(b, h, i, j + rotate_half_q_x1.shape()[3])] = rotate_half_q_x1[Index(b, h, i, j)]
                        rotate_half_k_out[Index(b, h, i, j + rotate_half_k_x1.shape()[3])] = rotate_half_k_x1[Index(b, h, i, j)]
        
        new_cos_2 = Tensor[DType.float32] (q.shape())
        new_sin_2 = Tensor[DType.float32] (q.shape())


        for i in range(new_cos_2.shape()[0]):
            for j in range(new_cos_2.shape()[1]):
                for k in range(new_cos_2.shape()[2]):
                    for l in range(new_cos_2.shape()[3]):
                        new_cos_2[Index(i,j,k,l)] = new_cos[Index(0,0,k,l)]
                        new_sin_2[Index(i,j,k,l)] = new_sin[Index(0,0,k,l)]
        
        q_embed = q*new_cos_2 + (rotate_half_q_out* new_sin_2)
        k_embed = (k * new_cos_2) + (rotate_half_k_out * new_sin_2)
        embs = List[Tensor[DType.float32]] ()
        embs.append(q_embed)
        embs.append(k_embed)
        return embs


fn chunk_qkv(qkv: Tensor[DType.float32], inout q: Tensor[DType.float32], inout start_qkv: Int):
    var start_q = 0
    alias load_sizes = 2048
    var q_num_elements = q.num_elements()
    for i in range(0, q_num_elements, load_sizes):
        q.store(start_q, qkv.load[width=load_sizes](start_qkv))
        start_q += load_sizes
        start_qkv += load_sizes*3

fn apply_rotary_pos_emb(q:Tensor[DType.float32], k:Tensor[DType.float32], cos:Tensor[DType.float32], 
                        sin:Tensor[DType.float32], position_ids:Tensor[DType.float32]) raises ->List[Tensor[DType.float32]]:
    new_cos = Tensor[DType.float32] (position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1])
    new_sin = Tensor[DType.float32] (position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1])
    for i in range(new_cos.shape()[0]):
        for j in range(new_cos.shape()[1]):
            pos_id = position_ids[i, j]
            for k in range(new_cos.shape()[2]):
                new_cos[Index(i,j,k)] = cos[Index(pos_id, k)]
                new_sin[Index(i,j,k)] = sin[Index(pos_id, k)]
    
    new_cos = new_cos.reshape((1,position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1]))
    new_sin = new_sin.reshape((1,position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1]))

    rotate_half_q_x1 = Tensor[DType.float32] (q.shape()[0], q.shape()[1], q.shape()[2], int(q.shape()[3]//2))
    rotate_half_q_x2 = Tensor[DType.float32] (q.shape()[0], q.shape()[1], q.shape()[2], int(q.shape()[3]//2))

    rotate_half_k_x1 = Tensor[DType.float32] (k.shape()[0], k.shape()[1], k.shape()[2], int(k.shape()[3]//2))
    rotate_half_k_x2 = Tensor[DType.float32] (k.shape()[0], k.shape()[1], k.shape()[2], int(k.shape()[3]//2))

    for b in range(rotate_half_q_x1.shape()[0]):
        for h in range(rotate_half_q_x1.shape()[1]):
            for i in range(rotate_half_q_x1.shape()[2]):
                for j in range(rotate_half_q_x1.shape()[3]):
                    # First half remains the same
                    rotate_half_q_x1[Index(b, h, i, j)] = q[Index(b, h, i, j)]
                    rotate_half_k_x1[Index(b, h, i, j)] = k[Index(b, h, i, j)]
                    # Second half is negated and swapped
                    rotate_half_q_x2[Index(b, h, i, j)] = -1 * q[Index(b, h, i, j + 16)]
                    rotate_half_k_x2[Index(b, h, i, j)] = -1 * k[Index(b, h, i, j + 16)]

    
    rotate_half_q_out = Tensor[DType.float32] (rotate_half_q_x1.shape()[0],rotate_half_q_x1.shape()[1],
                                             rotate_half_q_x1.shape()[2], rotate_half_q_x1.shape()[3] + rotate_half_q_x2.shape()[3])
    rotate_half_k_out = Tensor[DType.float32] (rotate_half_k_x1.shape()[0],rotate_half_k_x1.shape()[1],
                                             rotate_half_k_x1.shape()[2], rotate_half_k_x1.shape()[3] + rotate_half_k_x2.shape()[3])
    
    for b in range(rotate_half_q_x1.shape()[0]):
        for h in range(rotate_half_q_x1.shape()[1]):
            for i in range(rotate_half_q_x1.shape()[2]):
                for j in range(rotate_half_q_x1.shape()[3]):
                    # First 16 elements: negated -x2
                    rotate_half_q_out[Index(b, h, i, j)] = rotate_half_q_x2[Index(b, h, i, j)]
                    rotate_half_k_out[Index(b, h, i, j)] = rotate_half_k_x2[Index(b, h, i, j)]
                    # Next 16 elements: x1
                    rotate_half_q_out[Index(b, h, i, j + rotate_half_q_x1.shape()[3])] = rotate_half_q_x1[Index(b, h, i, j)]
                    rotate_half_k_out[Index(b, h, i, j + rotate_half_k_x1.shape()[3])] = rotate_half_k_x1[Index(b, h, i, j)]
    
    new_cos = Tensor[DType.float32] (q.shape())
    new_sin = Tensor[DType.float32] (q.shape())

    for i in range(new_cos.shape()[0]):
        for j in range(new_cos.shape()[1]):
            for k in range(new_cos.shape()[2]):
                for l in range(new_cos.shape()[3]):
                    new_cos[Index(i,j,k,l)] = cos[Index(k,l)]
                    new_sin[Index(i,j,k,l)] = sin[Index(k,l)]
    
    q_embed = q*new_cos + (rotate_half_q_out* new_sin)
    k_embed = (k * new_cos) + (rotate_half_k_out * new_sin)
    embs = List[Tensor[DType.float32]] ()
    embs.append(q_embed)
    embs.append(k_embed)
    return embs

fn scaled_dot_product_attention(query:Tensor[DType.float32], key:Tensor[DType.float32], value:Tensor[DType.float32],
                                transpose_21:Model, multiplication_4D:Model, addition_42:Model ,softmax:Model, 
                                multiplication_4D_2:Model, is_causal:Bool = False) raises -> Tensor[DType.float32]:
    
    var base = Float32(query.shape()[-1])
    var exponent = Float32(0.5)
    var scale_factor:Float32 = 1 / pow(base, exponent)

    # Initialize attn_bias based on query and key dimensions
    var L = query.shape()[-2]  # Sequence length of query
    var S = key.shape()[-2]    # Sequence length of key
    var attn_bias = Tensor[DType.float32]((L, S))

    # Apply causal masking only if is_causal is True
    if is_causal:
        for i in range(L):
            for j in range(S):
                if j > i:
                    attn_bias.store(Index(i, j), -inf[DType.float32]())

    # Transpose key and compute attention weights
    var results = transpose_21.execute("input0", key)
    var key_transpose = results.get[DType.float32]("output0")
    results = multiplication_4D.execute("input0", query, "input1", key_transpose)
    var atten_weights = results.get[DType.float32]("output0")
    var attention_weights = atten_weights * scale_factor

    # Apply the attention bias to the attention weights
    results = addition_42.execute("input0", attention_weights, "input1", attn_bias)
    attention_weights = results.get[DType.float32]("output0")

    # Softmax operation over the last dimension of attention weights
    var attn_weights = Tensor[DType.float32](attention_weights.shape())
    for i in range(attn_weights.shape()[0]):
        for j in range(attn_weights.shape()[1]):
            for k in range(attn_weights.shape()[2]):
                # Collect the entire last dimension (attention scores for each head for one sequence element)
                var new_tens = Tensor[DType.float32](attn_weights.shape()[3])  # Head dimension (S)

                # Copy each element from attention_weights into new_tens
                for l in range(attn_weights.shape()[3]):  # Head dimension
                    new_tens[l] = attention_weights[Index(i, j, k, l)]

                # Perform softmax on the extracted last dimension
                var results = softmax.execute("input0", new_tens)
                var xd = results.get[DType.float32]("output0")

                # Store the softmaxed values back into the attention weights
                for l in range(attn_weights.shape()[3]):
                    attn_weights[Index(i, j, k, l)] = xd[l]

    results = multiplication_4D_2.execute("input0", attn_weights, "input1", value)
    var output = results.get[DType.float32]("output0")

    return output



fn Gelu(x:Tensor[DType.float32], tanh:Model) raises -> Tensor[DType.float32]:
    # print(0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3.0)))))        
    var p = x*x*x
    var a = 0.044715 * p
    var m = x+a
    var m2 = pi_sqrt * m
    var results = tanh.execute("input0", m2)
    var tanh_out = results.get[DType.float32]("output0")
    plus = 1 + tanh_out
    result = 0.5*x*plus
    return result

fn main() raises:
    print("Compiling Graphs", end = " ")
    var session = engine.InferenceSession()

    var graph1 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m")))
    var transposed = ops.transpose(graph1[0],-1,-2)
    graph1.output(transposed)
    graph1.verify()
    var transpose = session.load(graph1)
    print(".", end = " ")

    var graph2 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "n")))
    var out2 = graph2[0] + graph2[1]
    graph2.output(out2)
    graph2.verify()
    var addition = session.load(graph2)

    var graph3 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "n","x")))
    var out3 = graph3[0] @ graph3[1]
    graph3.output(out3)
    graph3.verify()
    var multiplication = session.load(graph3)
    print(".", end = " ")

    var graph5 = Graph(in_types=List[Type](TensorType(DType.float32, 1, "b", "c")))
    var hollo = ops.mean(graph5[0])
    graph5.output(hollo)
    graph5.verify()
    var mean = session.load(graph5)

    var graph6 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d")))
    transposed = ops.transpose(graph6[0],1,2)
    graph6.output(transposed)
    graph6.verify()
    var transpose_12 = session.load(graph6)
    print(".", end = " ")

    var graph7 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d")))
    transposed = ops.transpose(graph7[0],-2,-1)
    graph7.output(transposed)
    graph7.verify()
    var transpose_21 = session.load(graph7)
    print(".", end = " ")

    var graph8 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d"), 
                                           TensorType(DType.float32, "a", "b", "d", "e")))
    var out8 = graph8[0] @ graph8[1]
    graph8.output(out8)
    graph8.verify()
    var multiplication_4D = session.load(graph8)
    print(".", end = " ")

    var graph9 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n", "o"), TensorType(DType.float32, "n", "o")))
    var out9 = graph9[0] + graph9[1]
    graph9.output(out9)
    graph9.verify()
    var addition_42 = session.load(graph9)
    print(".", end = " ")

    var graph10 = Graph(in_types=List[Type](TensorType(DType.float32, "a")))
    var softmaxed = ops.softmax(graph10[0])
    graph10.output(softmaxed)
    graph10.verify()
    var softmax = session.load(graph10)
    print(".", end = " ")

    var graph11 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "m", "n", "o"), 
                                           TensorType(DType.float32, "a", "m", "o", "x")))
    var out11 = graph11[0] @ graph11[1]
    graph11.output(out11)
    graph11.verify()
    var multiplication_4D_2 = session.load(graph11)
    print(".", end = " ")

    var graph12 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c")))
    var tanhed = ops.tanh(graph12[0])
    graph12.output(tanhed)
    graph12.verify()
    var tanh = session.load(graph12)
    print(".", end = " ")

    ###################################################################################################################
    print()
    print("Compiling LLM", end = " ")
    
    var mypython = Python.import_module("helper")
    var tensors = load("encoder_output.maxckpt")
    # var weights = load("text_model.maxckpt")
    var w:PythonObject = mypython.h7_state_dict()

    var encoder_output = tensors.get[DType.float32]("x")
    print(".", end = " ")

    cos_cache = numpy_to_tensor(mypython.cos_sin("cos_cached"))
    sin_cache = numpy_to_tensor(mypython.cos_sin("sin_cached"))
    print(".", end = " ")

    ln = List[LayerNorm] ()
    qkv_lin = List[Linear] ()
    outproj_lin = List[Linear] ()
    fc1_lin = List[Linear] ()
    fc2_lin = List[Linear] ()

    for i in range(0,8,1):

        ln_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.ln.weight'])
        ln_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.ln.bias'])
        ln.append(LayerNorm(ln_weight, ln_bias))
        print(".", end = " ")

        qkv_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.Wqkv.weight'])
        qkv_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.Wqkv.bias'])
        qkv_lin.append(Linear(qkv_weight, qkv_bias))
        print(".", end = " ")

        outproj_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.out_proj.weight'])
        outproj_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.out_proj.bias'])
        outproj_lin.append(Linear(outproj_weight, outproj_bias))
        print(".", end = " ")

        fc1_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc1.weight'])
        fc1_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc1.bias'])
        fc1_lin.append(Linear(fc1_weight, fc1_bias))
        print(".", end = " ")

        fc2_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc2.weight'])
        fc2_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc2.bias'])
        fc2_lin.append(Linear(fc2_weight, fc2_bias))
        print(".", end = " ")

    w = mypython.h8_state_dict()
    for i in range(8,18,1):

        ln_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.ln.weight'])
        ln_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.ln.bias'])
        ln.append(LayerNorm(ln_weight, ln_bias))
        print(".", end = " ")

        qkv_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.Wqkv.weight'])
        qkv_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.Wqkv.bias'])
        qkv_lin.append(Linear(qkv_weight, qkv_bias))
        print(".", end = " ")

        outproj_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.out_proj.weight'])
        outproj_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.out_proj.bias'])
        outproj_lin.append(Linear(outproj_weight, outproj_bias))
        print(".", end = " ")

        fc1_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc1.weight'])
        fc1_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc1.bias'])
        fc1_lin.append(Linear(fc1_weight, fc1_bias))
        print(".", end = " ")

        fc2_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc2.weight'])
        fc2_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc2.bias'])
        fc2_lin.append(Linear(fc2_weight, fc2_bias))
        print(".", end = " ")

    w = mypython.h18_state_dict()
    for i in range(18,24,1):

        ln_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.ln.weight'])
        ln_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.ln.bias'])
        ln.append(LayerNorm(ln_weight, ln_bias))
        print(".", end = " ")

        qkv_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.Wqkv.weight'])
        qkv_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.Wqkv.bias'])
        qkv_lin.append(Linear(qkv_weight, qkv_bias))
        print(".", end = " ")

        outproj_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.out_proj.weight'])
        outproj_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mixer.out_proj.bias'])
        outproj_lin.append(Linear(outproj_weight, outproj_bias))
        print(".", end = " ")

        fc1_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc1.weight'])
        fc1_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc1.bias'])
        fc1_lin.append(Linear(fc1_weight, fc1_bias))
        print(".", end = " ")

        fc2_weight = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc2.weight'])
        fc2_bias = numpy_to_tensor(w['transformer.h.'+str(i)+'.mlp.fc2.bias'])
        fc2_lin.append(Linear(fc2_weight, fc2_bias))
        print(".", end = " ")

    lm_head_ln_weight = numpy_to_tensor(w['lm_head.ln.weight'])
    lm_head_ln_bias = numpy_to_tensor(w['lm_head.ln.bias'])
    lm_head_lin_weight = numpy_to_tensor(w['lm_head.linear.weight'])
    lm_head_lin_bias = numpy_to_tensor(w['lm_head.linear.bias'])

    lm_head_ln = LayerNorm(lm_head_ln_weight, lm_head_ln_bias)
    lm_head_lin = Linear(lm_head_lin_weight, lm_head_lin_bias)

    emb_matrix = w['transformer.embd.wte.weight']

    w.__del__()

    while(1):
        values = List[Int] ()
        print()
        print("Running the model")
        py_builtins = Python.import_module("builtins")
        holla = py_builtins.input("Enter you question: ")
        var question = '\n\nQuestion: '+ str(holla) +' \n\nAnswer:'
        print(question)
        here = PythonObject()
        var input_len = 0
        var flag = True

        var past_key_states = List[Tensor[DType.float32]] ()
        var past_value_states = List[Tensor[DType.float32]] ()

        words = 0
        while(words <=128):
            inputs_embeds = Tensor[DType.float32] ()
            if words == 0:
                inputs_embeds = numpy_to_tensor(mypython.text_emb(question, tensor_to_numpy(encoder_output), emb_matrix))

            else:
                inputs_embeds = numpy_to_tensor(mypython.embedding_function(here, emb_matrix))


            input_to_layer = inputs_embeds

            if words == 0:
                input_len = input_to_layer.shape()[1]
                position_ids = Tensor[DType.float32] (1,input_len)
                count = 0
                for i in range(position_ids.shape()[0]):
                    for j in range(position_ids.shape()[1]):
                        position_ids[Index(i,j)] = count
                        count +=1
                flag = True
            else:
                position_ids = Tensor[DType.float32] (1,1)
                position_ids[Index(0,0)] = input_len
                input_len +=1
                flag = False

            for i in range(24):

                residual = input_to_layer
                var ln_out = ln[i].forward(input_to_layer, mean)

                bsz = ln_out.shape()[0]
                q_len = ln_out.shape()[1]

                qkv = qkv_lin[i].forward(ln_out, transpose, multiplication, addition)

                qkv_states = QKVstates(bsz, q_len, qkv)
                qkv_states_list = qkv_states.forward(transpose_12)
                query_states = qkv_states_list[0]
                key_states = qkv_states_list[1]
                value_states = qkv_states_list[2]

                cos = Tensor[DType.float32] (input_len, num_attention_heads)
                sin = Tensor[DType.float32] (input_len, num_attention_heads)

                for i in range(cos.shape()[0]):
                    for j in range(cos.shape()[1]):
                        cos[Index(i,j)] = cos_cache[Index(i,j)]
                        sin[Index(i,j)] = sin_cache[Index(i,j)]


                rot_pass = RotPass(query_states, key_states)
                qk_rot_pass = rot_pass.forward()
                query_rot = qk_rot_pass[0]
                query_pass = qk_rot_pass[1]
                key_rot =  qk_rot_pass[2]
                key_pass = qk_rot_pass[3]


                rot_pos_emb = RotPosEmb(cos, sin, position_ids)
                embs = rot_pos_emb.forward(query_rot, key_rot)
                query_rot = embs[0]
                key_rot = embs[1]

                new_query_states = Tensor[DType.float32] (query_pass.shape()[0], query_pass.shape()[1], query_pass.shape()[2], 
                                                        query_pass.shape()[3]+query_rot.shape()[3])
                new_key_states = Tensor[DType.float32] (key_pass.shape()[0], key_pass.shape()[1], key_pass.shape()[2], 
                                                        key_pass.shape()[3]+key_rot.shape()[3])
                for i in range(query_rot.shape()[0]):
                    for j in range(query_rot.shape()[1]):
                        for k in range(query_rot.shape()[2]):
                            for l in range(query_rot.shape()[3]):
                                new_query_states[Index(i,j,k,l)] = query_rot[Index(i, j, k, l)]
                                new_query_states[Index(i,j,k,l+query_rot.shape()[3])] = query_pass[Index(i, j, k, l)]

                                new_key_states[Index(i,j,k,l)] = key_rot[Index(i, j, k, l)]
                                new_key_states[Index(i,j,k,l+key_rot.shape()[3])] = key_pass[Index(i, j, k, l)]

                if words == 0:
                    past_key_states.append(new_key_states)
                    past_value_states.append(value_states)
                elif words != 0:
                    new_tens_keys = Tensor[DType.float32] (past_key_states[i].shape()[0], past_key_states[i].shape()[1],
                                                    past_key_states[i].shape()[2] + new_key_states.shape()[2], 
                                                    past_key_states[i].shape()[3])
                    new_tens_values = Tensor[DType.float32] (past_value_states[i].shape()[0], past_value_states[i].shape()[1],
                                                    past_value_states[i].shape()[2] + value_states.shape()[2], 
                                                    past_value_states[i].shape()[3])

                    for w in range(new_tens_keys.shape()[0]):
                        for x in range(new_tens_keys.shape()[1]):
                            for y in range(new_tens_keys.shape()[2]):
                                for z in range(new_tens_keys.shape()[3]):
                                    if y < past_key_states[i].shape()[2]:
                                        new_tens_keys[Index(w, x, y, z)] = past_key_states[i][Index(w, x, y, z)]
                                        new_tens_values[Index(w, x, y, z)] = past_value_states[i][Index(w, x, y, z)]
                                    else:
                                        new_tens_keys[Index(w, x, y, z)] = new_key_states[Index(w, x, 0, z)]
                                        new_tens_values[Index(w, x, y, z)] = value_states[Index(w, x, 0, z)]
                    
                    past_key_states[i] = new_tens_keys
                    past_value_states[i] = new_tens_values
                    
                    new_key_states = new_tens_keys
                    value_states = new_tens_values
                
                attn_output = scaled_dot_product_attention(new_query_states, new_key_states, value_states, transpose_21, multiplication_4D, 
                                                        addition_42, softmax, multiplication_4D_2, flag)

                results = transpose_12.execute("input0", attn_output)
                var attn_output_t = results.get[DType.float32]("output0")
                attn_output_r = attn_output_t.reshape((bsz, q_len, hidden_size))
                
        

                attention_output = outproj_lin[i].forward(attn_output_r, transpose, multiplication, addition)

                fc1_out = fc1_lin[i].forward(ln_out, transpose, multiplication, addition)
                
                gelu_out = Gelu(fc1_out, tanh)

                fc2_out = fc2_lin[i].forward(gelu_out, transpose, multiplication, addition)

                hidden_states = attention_output + fc2_out + residual

                input_to_layer = hidden_states

            j_index = input_to_layer.shape()[1]

            new = Tensor[DType.float32] (1,1,hidden_size)
            for i in range(1):
                for j in range(1):
                    for k in range(hidden_size):
                        new[Index(i,j,k)] = input_to_layer[Index(0,j_index-1,k)]
            
            lm_ln = lm_head_ln.forward(new, mean)
            lm_lin = lm_head_lin.forward(lm_ln, transpose, multiplication, addition)
            

            here = mypython.argmax_index(tensor_to_numpy(lm_lin))
            
            values.append(here[0][0])

            if here[0][0] == 50256:
                break
            input_to_layer = lm_lin
            
            words +=1
        
        var np = Python.import_module("numpy")
        var np_values = np.zeros((1, len(values)), np.int32)
        for i in range(np_values.shape[0]):
            for j in range(np_values.shape[1]):
                np_values[i][j] = values[j]
        
        output = mypython.decode(np_values)
        print(output)

