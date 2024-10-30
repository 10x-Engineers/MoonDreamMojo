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
from algorithm import vectorize
from max.graph.type import Dim


alias num_attention_heads = 32
alias hidden_size = 2048
alias head_dim = hidden_size // num_attention_heads
alias min_len = 700
alias pi_sqrt = 0.7978845608028654
alias batch_size = 1
alias scale_fac = 0.125
alias nelts = 32


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
    
    fn forward(self, inputs_embeds: Tensor[DType.float32], layer_norm_: Model, norm:Model) 
               raises -> Tensor[DType.float32]:
        results = layer_norm_.execute("input0", inputs_embeds, "input1", self.w, "input2", self.b)
        out = results.get[DType.float32]("output0")
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
    
    fn forward(self, inputs_mat: Tensor[DType.float32], model: Model) 
               raises -> Tensor[DType.float32]:
        results = model.execute("input0", inputs_mat, "input1", self.w, "input2", self.b)
        output = results.get[DType.float32]("output0")
        return output

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

    fn forward(self, transpose_12: Model, model:Model) raises -> List[Tensor[DType.float32]]:
        out = List[Tensor[DType.float32]] ()
        
        results = model.execute("input0", self.qkv)
        query_states = results.get[DType.float32]("output0")
        key_states = results.get[DType.float32]("output1")
        value_states = results.get[DType.float32]("output2")

        reshaped_q = query_states.reshape((self.bsz, self.q_len, num_attention_heads, head_dim))
        results = transpose_12.execute("input0", reshaped_q)
        query_states = results.get[DType.float32]("output0")

        reshaped_k = key_states.reshape((self.bsz, self.q_len, num_attention_heads, head_dim))
        results = transpose_12.execute("input0", reshaped_k)
        key_states = results.get[DType.float32]("output0")

        reshaped_v = value_states.reshape((self.bsz, self.q_len, num_attention_heads, head_dim))
        results = transpose_12.execute("input0", reshaped_v)
        value_states = results.get[DType.float32]("output0")

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
        out = List[Tensor[DType.float32]] ()
        
        query_rot = Tensor[DType.float32] (self.query_states.shape()[0], self.query_states.shape()[1], 
                                    self.query_states.shape()[2], num_attention_heads)
        @parameter
        fn store_tensor(index: Int):
            i = index // (query_rot.shape()[1] * query_rot.shape()[2])
            j = (index // query_rot.shape()[2]) % query_rot.shape()[1]
            k = index % query_rot.shape()[2]
            query_rot.store(Index(i, j, k, 0), 
                            self.query_states.load[width=num_attention_heads] (Index(i, j, k, 0)))
        total_elements = query_rot.shape()[0] * query_rot.shape()[1] * query_rot.shape()[2]
        parallelize[store_tensor] (total_elements)

        query_pass = Tensor[DType.float32] (self.query_states.shape()[0], self.query_states.shape()[1], 
                                self.query_states.shape()[2], num_attention_heads)
        @parameter
        fn store_query_pass(index: Int):
            i = index // (query_pass.shape()[1] * query_pass.shape()[2])
            j = (index // query_pass.shape()[2]) % query_pass.shape()[1]
            k = index % query_pass.shape()[2]
            query_pass.store(Index(i, j, k, 0), 
                            self.query_states.load[width=num_attention_heads](Index(i, j, k, num_attention_heads)))
        total_elements = query_pass.shape()[0] * query_pass.shape()[1] * query_pass.shape()[2]
        parallelize[store_query_pass] (total_elements)

        key_rot = Tensor[DType.float32] (self.key_states.shape()[0], self.key_states.shape()[1], 
                                self.key_states.shape()[2], num_attention_heads)
        @parameter
        fn store_key_rot(index: Int):
            i = index // (key_rot.shape()[1] * key_rot.shape()[2])
            j = (index // key_rot.shape()[2]) % key_rot.shape()[1]
            k = index % key_rot.shape()[2]
            key_rot.store(Index(i, j, k, 0), 
                        self.key_states.load[width=num_attention_heads](Index(i, j, k, 0)))
        total_elements_key_rot = key_rot.shape()[0] * key_rot.shape()[1] * key_rot.shape()[2]
        parallelize[store_key_rot] (total_elements_key_rot)

        key_pass = Tensor[DType.float32] (self.key_states.shape()[0], self.key_states.shape()[1], 
                                self.key_states.shape()[2], num_attention_heads)
        @parameter
        fn store_key_pass(index: Int):
            i = index // (key_pass.shape()[1] * key_pass.shape()[2])
            j = (index // key_pass.shape()[2]) % key_pass.shape()[1]
            k = index % key_pass.shape()[2]
            key_pass.store(Index(i, j, k, 0), 
                        self.key_states.load[width=num_attention_heads](Index(i, j, k, num_attention_heads)))

        total_elements_key_pass = key_pass.shape()[0] * key_pass.shape()[1] * key_pass.shape()[2]
        parallelize[store_key_pass] (total_elements_key_pass)
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
    
    fn forward(self, q:Tensor[DType.float32], k:Tensor[DType.float32], layer:Int, model:Model, wtf1:Model, session:engine.InferenceSession)
               raises ->List[Tensor[DType.float32]]:
        if layer == 0:
            new_cos = self.cos
            new_cos = new_cos.reshape((1,1,self.pos_ids.shape()[1],self.cos.shape()[1]))
            new_sin = self.sin
            new_sin = new_sin.reshape((1,1,self.pos_ids.shape()[1],self.cos.shape()[1]))

            rotate_half_q_x1 = Tensor[DType.float32] (q.shape()[0], q.shape()[1], q.shape()[2], int(q.shape()[3] // 2))
            rotate_half_q_x2 = Tensor[DType.float32] (q.shape()[0], q.shape()[1], q.shape()[2], int(q.shape()[3] // 2))
            rotate_half_k_x1 = Tensor[DType.float32] (k.shape()[0], k.shape()[1], k.shape()[2], int(k.shape()[3] // 2))
            rotate_half_k_x2 = Tensor[DType.float32] (k.shape()[0], k.shape()[1], k.shape()[2], int(k.shape()[3] // 2))
            for b in range(rotate_half_q_x1.shape()[0]):
                for h in range(rotate_half_q_x1.shape()[1]):
                    for i in range(rotate_half_q_x1.shape()[2]):
                        # Vectorized loading of the first half
                        rotate_half_q_x1.store(Index(b, h, i, 0), q.load[width=16](Index(b, h, i, 0)))
                        rotate_half_k_x1.store(Index(b, h, i, 0), k.load[width=16](Index(b, h, i, 0)))                        
                        # Vectorized loading of the second half with negation
                        rotate_half_q_x2.store(Index(b, h, i, 0), -1 * q.load[width=16](Index(b, h, i, 16)))
                        rotate_half_k_x2.store(Index(b, h, i, 0), -1 * k.load[width=16](Index(b, h, i, 16)))
            
            rotate_half_q_out = Tensor[DType.float32] (rotate_half_q_x1.shape()[0], rotate_half_q_x1.shape()[1], 
                                rotate_half_q_x1.shape()[2], rotate_half_q_x1.shape()[3] + rotate_half_q_x2.shape()[3])
            rotate_half_k_out = Tensor[DType.float32] (rotate_half_k_x1.shape()[0], rotate_half_k_x1.shape()[1], 
                                rotate_half_k_x1.shape()[2], rotate_half_k_x1.shape()[3] + rotate_half_k_x2.shape()[3])
            for b in range(rotate_half_q_x1.shape()[0]):
                for h in range(rotate_half_q_x1.shape()[1]):
                    for i in range(rotate_half_q_x1.shape()[2]):
                        # Vectorized storing of the first half: -x2
                        rotate_half_q_out.store(Index(b, h, i, 0), rotate_half_q_x2.load[width=16](Index(b, h, i, 0)))
                        rotate_half_k_out.store(Index(b, h, i, 0), rotate_half_k_x2.load[width=16](Index(b, h, i, 0)))
                        # Vectorized storing of the second half: x1
                        rotate_half_q_out.store(Index(b, h, i, rotate_half_q_x1.shape()[3]), 
                                                rotate_half_q_x1.load[width=16](Index(b, h, i, 0)))
                        rotate_half_k_out.store(Index(b, h, i, rotate_half_k_x1.shape()[3]), 
                                                rotate_half_k_x1.load[width=16](Index(b, h, i, 0)))

            var inputs_to_tm_x = session.new_tensor_map()
            inputs_to_tm_x.borrow("input0", q)
            inputs_to_tm_x.borrow("input1", k)
            inputs_to_tm_x.borrow("input2", new_cos)
            inputs_to_tm_x.borrow("input3", new_sin)
            inputs_to_tm_x.borrow("input4", rotate_half_q_out)
            inputs_to_tm_x.borrow("input5", rotate_half_k_out)

            results = model.execute(inputs_to_tm_x)
            q_embed = results.get[DType.float32]("output0")
            k_embed = results.get[DType.float32]("output1")

            print(q.shape(), k.shape(), new_cos.shape(), new_sin.shape(), rotate_half_q_out.shape(), rotate_half_k_out.shape())

            embs = List[Tensor[DType.float32]] ()
            embs.append(q_embed)
            embs.append(k_embed)
            return embs
        
        else:
            new_cos = Tensor[DType.float32] (self.pos_ids.shape()[0], self.pos_ids.shape()[1], self.cos.shape()[1])
            new_sin = Tensor[DType.float32] (self.pos_ids.shape()[0], self.pos_ids.shape()[1], self.cos.shape()[1])
            for i in range(new_cos.shape()[0]):
                for j in range(new_cos.shape()[1]):
                    pos_id = self.pos_ids[i, j]
                    for k in range(0, new_cos.shape()[2], num_attention_heads):
                        new_cos.store(Index(i, j, k), self.cos.load[width=num_attention_heads](Index(pos_id, k)))
                        new_sin.store(Index(i, j, k), self.sin.load[width=num_attention_heads](Index(pos_id, k)))
            new_cos = new_cos.reshape((1,self.pos_ids.shape()[0], self.pos_ids.shape()[1], self.cos.shape()[1]))
            new_sin = new_sin.reshape((1,self.pos_ids.shape()[0], self.pos_ids.shape()[1], self.cos.shape()[1]))


            var inputs_to_tm = session.new_tensor_map()
            inputs_to_tm.borrow("input0", q)
            inputs_to_tm.borrow("input1", k)
            inputs_to_tm.borrow("input2", new_cos)
            inputs_to_tm.borrow("input3", new_sin)
            results = wtf1.execute(inputs_to_tm)
            q_embed = results.get[DType.float32]("output0")
            k_embed = results.get[DType.float32]("output1")

            print(q.shape(), k.shape(), new_cos.shape(), new_sin.shape())
            embs = List[Tensor[DType.float32]] ()
            embs.append(q_embed)
            embs.append(k_embed)
            return embs

fn main() raises:
    print("Compiling Graphs", end = " ")
    var session = engine.InferenceSession()

    var graph6 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d")))
    transposed = ops.transpose(graph6[0],1,2)
    graph6.output(transposed)
    graph6.verify()
    var transpose_12 = session.load(graph6)
    print(".", end = " ")

    var fc_2 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c"), 
                     TensorType(DType.float32, "x", "c"), TensorType(DType.float32, "x")))
    w_t = ops.transpose(fc_2[1],-1,-2)
    mu = fc_2[0] @ w_t
    addi = mu + fc_2[2]
    fc_2.output(addi)
    fc_2.verify()
    var lin_new = session.load(fc_2)
    print(".", end = " ")

    var qkv_model = Graph(in_types=List[Type](TensorType(DType.float32, 1, "x", 6144)))
    var q = qkv_model[0][0:1, :, 0:2048]
    var k = qkv_model[0][0:1, :, 2048:4096]
    var v = qkv_model[0][0:1, :, 4096:6144]
    var o = List[Symbol] ()
    o.append(q)
    o.append(k)
    o.append(v)
    qkv_model.output(o)
    qkv_model.verify()
    var qkv_ = session.load(qkv_model)
    print(".", end = " ")

    var in_types_ = List[Type] (TensorType(DType.float32, 1, 32, 1, 32), TensorType(DType.float32, 1, 32, 1, 32), 
                                TensorType(DType.float32, 1, 32, 1, 32), TensorType(DType.float32, 1, 32, 1, 32))
    var concat_ = Graph(in_types=in_types_)
    var inputs1 = List[Symbol] (concat_[0], concat_[1])
    var inputs2 = List[Symbol] (concat_[2], concat_[3])
    var c1 = ops.concat(inputs1, 3)
    var c2 = ops.concat(inputs2, 3)
    var concat_out_ = List[Symbol] ()
    concat_out_.append(c1)
    concat_out_.append(c2)
    concat_.output(concat_out_)
    concat_.verify()
    var concat_graph = session.load(concat_)
    print(".", end = " ")

    var xyz_0 = Graph(in_types=List[Type](TensorType(DType.float32, 1, 32, "x", 64), TensorType(DType.float32, 1, 32, "x", 64), 
                                        TensorType(DType.float32, 1, 32, "x", 64), TensorType(DType.float32, "x", "x")))
    var key_transposed_0 = ops.transpose(xyz_0[1],-2,-1)
    var mul_0 = xyz_0[0] @ key_transposed_0
    var scaleing_factor_0 = mul_0 * scale_fac
    var attention_weights_0 = scaleing_factor_0 + xyz_0[3]
    var softmax_applied_0 = ops.softmax(attention_weights_0)
    var attention_out_0 = softmax_applied_0 @ xyz_0[2]
    var atten_out_t_0 = ops.transpose(attention_out_0,1,2)
    xyz_0.output(atten_out_t_0)
    xyz_0.verify()
    var xyz_0_graph = session.load(xyz_0)
    print(".", end = " ")

    var xyz2_0 = Graph(in_types=List[Type](TensorType(DType.float32, 1, "x", 2048), TensorType(DType.float32, 1, "x", 2048),
                                        TensorType(DType.float32, 2048, 2048), TensorType(DType.float32, 2048),
                                        TensorType(DType.float32, 8192, 2048), TensorType(DType.float32, 8192),
                                        TensorType(DType.float32, 2048, 8192), TensorType(DType.float32, 2048),
                                        TensorType(DType.float32, 1, "x", 2048)))
    
    w_t = ops.transpose(xyz2_0[2],-1,-2)
    mu = xyz2_0[0] @ w_t
    attention_output_g = mu + xyz2_0[3]

    w_t = ops.transpose(xyz2_0[4],-1,-2)
    mu = xyz2_0[1] @ w_t
    fc1_out_g = mu + xyz2_0[5]

    var product_0 = fc1_out_g * fc1_out_g * fc1_out_g
    var a_const_0 = 0.044715 * product_0
    var m2_const_0 = fc1_out_g + a_const_0
    var m3_const_0 = pi_sqrt * m2_const_0
    var tanh_output_0 = ops.tanh(m3_const_0)
    plus_const_0 = 1 + tanh_output_0
    gelu_result = 0.5 * fc1_out_g * plus_const_0

    w_t = ops.transpose(xyz2_0[6],-1,-2)
    mu = gelu_result @ w_t
    fc2_out_g = mu + xyz2_0[7]

    hidden_states_g = attention_output_g + fc2_out_g + xyz2_0[8]
    xyz2_0.output(hidden_states_g)
    xyz2_0.verify()
    var xyz2_0_graph = session.load(xyz2_0)
    print(".", end = " ")

    var xyz = Graph(in_types=List[Type](TensorType(DType.float32, 1, 32, "x", 64), TensorType(DType.float32, 1, 32, "y", 64), 
                                        TensorType(DType.float32, 1, 32, "y", 64)))
    var key_transposed = ops.transpose(xyz[1],-2,-1)
    var mul = xyz[0] @ key_transposed
    var scaleing_factor = mul * scale_fac
    var softmax_applied = ops.softmax(scaleing_factor)
    var attention_out = softmax_applied @ xyz[2]
    var atten_out_t = ops.transpose(attention_out,1,2)
    xyz.output(atten_out_t)
    xyz.verify()
    var xyz_graph = session.load(xyz)
    print(".", end = " ")

    var xyz2 = Graph(in_types=List[Type](TensorType(DType.float32, 1, 1, 2048), TensorType(DType.float32, 1, 1, 2048),
                                        TensorType(DType.float32, 2048, 2048), TensorType(DType.float32, 2048),
                                        TensorType(DType.float32, 8192, 2048), TensorType(DType.float32, 8192),
                                        TensorType(DType.float32, 2048, 8192), TensorType(DType.float32, 2048),
                                        TensorType(DType.float32, 1, 1, 2048)))
    
    w_t = ops.transpose(xyz2[2],-1,-2)
    mu = xyz2[0] @ w_t
    attention_output_g = mu + xyz2[3]

    w_t = ops.transpose(xyz2[4],-1,-2)
    mu = xyz2[1] @ w_t
    fc1_out_g = mu + xyz2[5]

    var product = fc1_out_g * fc1_out_g * fc1_out_g
    var a_const = 0.044715 * product
    var m2_const = fc1_out_g + a_const
    var m3_const = pi_sqrt * m2_const
    var tanh_output = ops.tanh(m3_const)
    plus_const = 1 + tanh_output
    gelu_result = 0.5 * fc1_out_g * plus_const

    w_t = ops.transpose(xyz2[6],-1,-2)
    mu = gelu_result @ w_t
    fc2_out_g = mu + xyz2[7]

    hidden_states_g = attention_output_g + fc2_out_g + xyz2[8]
    xyz2.output(hidden_states_g)
    xyz2.verify()
    var xyz2_graph = session.load(xyz2)
    print(".", end = " ")

    var start_ = Graph(in_types=List[Type](TensorType(DType.float32, 1, 1, 2048), TensorType(DType.float32,2048), 
                                           TensorType(DType.float32,2048), TensorType(DType.float32, 6144, 2048), 
                                           TensorType(DType.float32, 6144)))
    
    var layer_norm = ops.layer_norm(start_[0],gamma = start_[1], beta = start_[2] , epsilon = 1e-5)

    var weight_t = ops.transpose(start_[3],-1,-2)
    var multiply_mat = layer_norm @ weight_t
    # var qkv_linear_out = (multiply_mat + start_[4]).reshape(1,1,6144)
    var qkv_linear_out = (multiply_mat + start_[4])
    var query = qkv_linear_out[0:1, 0:1, 0:2048].reshape(1, 1, num_attention_heads, head_dim)
    var key = qkv_linear_out[0:1, 0:1, 2048:4096].reshape(1, 1, num_attention_heads, head_dim)
    var val = qkv_linear_out[0:1, 0:1, 4096:6144].reshape(1, 1, num_attention_heads, head_dim)
    var query_trans = ops.transpose(query,1,2)
    var key_trans = ops.transpose(key,1,2)
    var val_trans = ops.transpose(val,1,2)

    var query1_rot_ = query_trans[0:1, 0:32, 0:1, 0:32]
    var query1_pass_ = query_trans[0:1, 0:32, 0:1, 32:64]
    var key1_rot_ = key_trans[0:1, 0:32, 0:1, 0:32]
    var key1_pass_ = key_trans[0:1, 0:32, 0:1, 32:64]
    var start_out = List[Symbol] ()

    start_out.append(val_trans)
    start_out.append(query1_rot_)
    start_out.append(query1_pass_)
    start_out.append(key1_rot_)
    start_out.append(key1_pass_)
    start_out.append(layer_norm)

    start_.output(start_out)
    start_.verify()
    var start_model = session.load(start_)
    print(".", end = " ")
    
    var head_ = Graph(in_types=List[Type](TensorType(DType.float32, 1, 1, 2048), TensorType(DType.float32,2048), 
                                           TensorType(DType.float32,2048), TensorType(DType.float32, 51200, 2048), 
                                           TensorType(DType.float32, 51200)))
    var head_norm = ops.layer_norm(head_[0],gamma = head_[1], beta = head_[2] , epsilon = 1e-5)

    var head_weight_t = ops.transpose(head_[3],-1,-2)
    var head_multiply_mat = head_norm @ head_weight_t
    var head_out = (head_multiply_mat + head_[4])

    head_.output(head_out)
    head_.verify()
    var head_model = session.load(head_)
    print(".", end = " ")

    var epsilon:Float32 = 1e-5
    var d = List[Dim] ()
    d.append(1)
    d.append("x")
    d.append(2048)
    var testing = Graph(in_types=List[Type](TensorType(DType.float32, 1, "x", 2048), TensorType(DType.float32, 2048), 
                                            TensorType(DType.float32, 2048)))
    var mean_ = ops.mean(testing[0])
    var mean_broadcasted = ops.broadcast_to(mean_, d)
    var diff = (testing[0] - mean_broadcasted)
    var squared_diff = diff ** 2
    var variance = ops.mean(squared_diff)
    var epsilon_symbol = testing.scalar(epsilon)
    var y = ops.add(variance, epsilon_symbol) ** 0.5
    var z = ops.broadcast_to(y, d)
    var normalized_states = diff/z
    var haha = (normalized_states * testing[1]) + testing[2]
    testing.output(haha)
    testing.verify()
    var layer_norm_ = session.load(testing)

    var dms = List[Dim] ()
    dms.append(1)
    dms.append(32)
    dms.append("x")
    dms.append(32)

    var rot_pos_emb_1 = Graph(in_types=List[Type](TensorType(DType.float32, 1, 32, 1, 32), TensorType(DType.float32,1, 32, 1, 32), 
                                           TensorType(DType.float32, 1, 1, "x", 32), TensorType(DType.float32, 1, 1, "x", 32)))
    var first_half_q1 = rot_pos_emb_1[0][0:1, 0:32, 0:1, 0:16]
    var first_half_k1 = rot_pos_emb_1[1][0:1, 0:32, 0:1, 0:16]
    var second_half_q1 = rot_pos_emb_1[0][0:1, 0:32, 0:1, 16:32] * -1
    var second_half_k1 = rot_pos_emb_1[1][0:1, 0:32, 0:1, 16:32] * -1
    var inputs_q1 = List[Symbol] (second_half_q1, first_half_q1)
    var inputs_k1 = List[Symbol] (second_half_k1, first_half_k1)
    var rotate_half_q_out1 = ops.concat(inputs_q1, 3)
    var rotate_half_k_out1 = ops.concat(inputs_k1, 3)
    var new_cos_2 = ops.broadcast_to(rot_pos_emb_1[2], dms)
    var new_sin_2 = ops.broadcast_to(rot_pos_emb_1[3], dms)
    var q_embed = (rot_pos_emb_1[0] * new_cos_2) + (rotate_half_q_out1 * new_sin_2)
    var k_embed = (rot_pos_emb_1[1] * new_cos_2) + (rotate_half_k_out1 * new_sin_2)
    var out = List[Symbol] ()
    out.append(q_embed)
    out.append(k_embed)
    rot_pos_emb_1.output(out)
    rot_pos_emb_1.verify()
    var wtf1 = session.load(rot_pos_emb_1)
    print(".", end = " ")

    var rot_pos_emb_0 = Graph(in_types=List[Type](TensorType(DType.float32, 1, 32, "x", 32), TensorType(DType.float32,1, 32, "x", 32), 
                                           TensorType(DType.float32, 1, 1, "x", 32), TensorType(DType.float32, 1, 1, "x", 32), 
                                           TensorType(DType.float32, 1, 32, "x", 32), TensorType(DType.float32,1, 32, "x", 32))) 
    var new_cos_20 = ops.broadcast_to(rot_pos_emb_0[2], dms)
    var new_sin_20 = ops.broadcast_to(rot_pos_emb_0[3], dms)
    var q_embed0 = (rot_pos_emb_0[0] * new_cos_20) + (rot_pos_emb_0[4] * new_sin_20)
    var k_embed0 = (rot_pos_emb_0[1] * new_cos_20) + (rot_pos_emb_0[5] * new_sin_20)
    var out0 = List[Symbol] ()
    out0.append(q_embed0)
    out0.append(k_embed0)
    rot_pos_emb_0.output(out0)
    rot_pos_emb_0.verify()
    var wtf0 = session.load(rot_pos_emb_0)
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
        first_token_time = 0.0
        start = now()
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
                var ln_out = Tensor[DType.float32]()
                if words == 0:
                    residual = input_to_layer
                    results = layer_norm_.execute("input0", input_to_layer, "input1", ln[i].w, "input2", ln[i].b)
                    ln_out = results.get[DType.float32]("output0")
                    bsz = ln_out.shape()[0]
                    q_len = ln_out.shape()[1]

                    qkv = qkv_lin[i].forward(ln_out, lin_new)

                    qkv_states = QKVstates(bsz, q_len, qkv)
                    qkv_states_list = qkv_states.forward(transpose_12, qkv_)
                    query_states = qkv_states_list[0]
                    key_states = qkv_states_list[1]
                    value_states = qkv_states_list[2]

                    rot_pass = RotPass(query_states, key_states)
                    qk_rot_pass = rot_pass.forward()
                    query_rot = qk_rot_pass[0]
                    query_pass = qk_rot_pass[1]
                    key_rot =  qk_rot_pass[2]
                    key_pass = qk_rot_pass[3]
                
                else:
                    residual = input_to_layer
                    var inputs_to_tm3 = session.new_tensor_map()
                    inputs_to_tm3.borrow("input0", input_to_layer)
                    inputs_to_tm3.borrow("input1", ln[i].w)
                    inputs_to_tm3.borrow("input2", ln[i].b)
                    inputs_to_tm3.borrow("input3", qkv_lin[i].w)
                    inputs_to_tm3.borrow("input4", qkv_lin[i].b)
                    results = start_model.execute(inputs_to_tm3)
                    value_states = results.get[DType.float32]("output0")
                    query_rot = results.get[DType.float32]("output1")
                    query_pass = results.get[DType.float32]("output2")
                    key_rot = results.get[DType.float32]("output3")
                    key_pass = results.get[DType.float32]("output4")
                    ln_out = results.get[DType.float32]("output5")

                    print(input_to_layer.shape(), ln[i].w.shape(), ln[i].b.shape(), qkv_lin[i].w.shape(), qkv_lin[i].b.shape())

                    bsz = ln_out.shape()[0]
                    q_len = ln_out.shape()[1]

                cos = Tensor[DType.float32] (input_len, num_attention_heads)
                sin = Tensor[DType.float32] (input_len, num_attention_heads)
                for i in range(cos.shape()[0]):
                    cos.store(Index(i, 0), cos_cache.load[width=num_attention_heads](Index(i, 0)))
                    sin.store(Index(i, 0), sin_cache.load[width=num_attention_heads](Index(i, 0)))

                if words == 0:
                    rot_pos_emb = RotPosEmb(cos, sin, position_ids)
                    embs = rot_pos_emb.forward(query_rot, key_rot, words, wtf0, wtf1, session)
                    query_rot = embs[0]
                    key_rot = embs[1]
                else:
                    rot_pos_emb = RotPosEmb(cos, sin, position_ids)
                    embs = rot_pos_emb.forward(query_rot, key_rot, words, wtf0, wtf1, session)
                    query_rot = embs[0]
                    key_rot = embs[1]
                    # new_cos = Tensor[DType.float32] (position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1])
                    # new_sin = Tensor[DType.float32] (position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1])
                    # for i in range(new_cos.shape()[0]):
                    #     for j in range(new_cos.shape()[1]):
                    #         pos_id = position_ids[i, j]
                    #         for k in range(0, new_cos.shape()[2], num_attention_heads):
                    #             new_cos.store(Index(i, j, k), cos.load[width=num_attention_heads](Index(pos_id, k)))
                    #             new_sin.store(Index(i, j, k), sin.load[width=num_attention_heads](Index(pos_id, k)))
                    # new_cos = new_cos.reshape((1,position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1]))
                    # new_sin = new_sin.reshape((1,position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1]))

                    # var inputs_to_tm_y = session.new_tensor_map()
                    # inputs_to_tm_y.borrow("input0", query_rot)
                    # inputs_to_tm_y.borrow("input1", key_rot)
                    # inputs_to_tm_y.borrow("input2", new_cos)
                    # inputs_to_tm_y.borrow("input3", new_sin)
                    # results = wtf1.execute(inputs_to_tm_y)
                    # query_rot = results.get[DType.float32]("output0")
                    # key_rot = results.get[DType.float32]("output1")

                    # print(query_rot.shape(), key_rot.shape(), new_cos.shape(), new_sin.shape())

                new_query_states = Tensor[DType.float32] (query_pass.shape()[0], query_pass.shape()[1], query_pass.shape()[2], 
                                                        query_pass.shape()[3]+query_rot.shape()[3])
                new_key_states = Tensor[DType.float32] (key_pass.shape()[0], key_pass.shape()[1], key_pass.shape()[2], 
                                                        key_pass.shape()[3]+key_rot.shape()[3])
                if words == 0:
                    for i in range(query_rot.shape()[0]):
                        for j in range(query_rot.shape()[1]):
                            for k in range(query_rot.shape()[2]):
                                for l in range(0, query_rot.shape()[3], 32):
                                    new_query_states.store(Index(i, j, k, l), query_rot.load[width=32](Index(i, j, k, l)))
                                    new_query_states.store(Index(i, j, k, l + query_rot.shape()[3]), query_pass.load[width=32](Index(i, j, k, l)))
                                    new_key_states.store(Index(i, j, k, l), key_rot.load[width=32](Index(i, j, k, l)))
                                    new_key_states.store(Index(i, j, k, l + key_rot.shape()[3]), key_pass.load[width=32](Index(i, j, k, l)))
                else:
                    var inputs_to_tm5 = session.new_tensor_map()
                    inputs_to_tm5.borrow("input0", query_rot)
                    inputs_to_tm5.borrow("input1", query_pass)
                    inputs_to_tm5.borrow("input2", key_rot)
                    inputs_to_tm5.borrow("input3", key_pass)

                    results = concat_graph.execute(inputs_to_tm5)
                    new_query_states = results.get[DType.float32]("output0")
                    new_key_states = results.get[DType.float32]("output1")

                    print(query_rot.shape(), query_pass.shape(), key_rot.shape(),  key_pass.shape())

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
                                if y < past_key_states[i].shape()[2]:
                                    new_tens_keys.store(Index(w, x, y, 0), 
                                                        past_key_states[i].load[width=64](Index(w, x, y, 0)))
                                    new_tens_values.store(Index(w, x, y, 0), 
                                                        past_value_states[i].load[width=64](Index(w, x, y, 0)))
                                else:
                                    new_tens_keys.store(Index(w, x, y, 0), 
                                                        new_key_states.load[width=64](Index(w, x, 0, 0)))
                                    new_tens_values.store(Index(w, x, y, 0),
                                                        value_states.load[width=64](Index(w, x, 0, 0)))
                    
                    past_key_states[i] = new_tens_keys
                    past_value_states[i] = new_tens_values
                    new_key_states = new_tens_keys
                    value_states = new_tens_values

                if words == 0:
                    var inputs_to_tm = session.new_tensor_map()
                    var L = new_query_states.shape()[-2]
                    var S = new_key_states.shape()[-2]
                    var attn_bias = Tensor[DType.float32]((L, S))
                    for i in range(L):
                        for j in range(S):
                            if j > i:
                                attn_bias.store(Index(i, j), -inf[DType.float32]())
                    
                    inputs_to_tm.borrow("input0", new_query_states)
                    inputs_to_tm.borrow("input1", new_key_states)
                    inputs_to_tm.borrow("input2", value_states)
                    inputs_to_tm.borrow("input3", attn_bias)
                    
                    results = xyz_0_graph.execute(inputs_to_tm)
                    var attn_output_t = results.get[DType.float32]("output0")
                    print(new_query_states.shape(), new_key_states.shape(), value_states.shape(), attn_bias.shape())

                    attn_output_r = attn_output_t.reshape((bsz, q_len, hidden_size))
                    
                    var inputs_to_tm2 = session.new_tensor_map()
                    inputs_to_tm2.borrow("input0", attn_output_r)
                    inputs_to_tm2.borrow("input1", ln_out)
                    inputs_to_tm2.borrow("input2", outproj_lin[i].w)
                    inputs_to_tm2.borrow("input3", outproj_lin[i].b)
                    inputs_to_tm2.borrow("input4", fc1_lin[i].w)
                    inputs_to_tm2.borrow("input5", fc1_lin[i].b)
                    inputs_to_tm2.borrow("input6", fc2_lin[i].w)
                    inputs_to_tm2.borrow("input7", fc2_lin[i].b)
                    inputs_to_tm2.borrow("input8", residual)

                    results = xyz2_0_graph.execute(inputs_to_tm2)
                    hidden_states = results.get[DType.float32]("output0")

                    print(attn_output_r.shape(), ln_out.shape(), outproj_lin[i].w.shape(), outproj_lin[i].b.shape(),
                          fc1_lin[i].w.shape(), fc1_lin[i].b.shape(), fc2_lin[i].w.shape(), fc2_lin[i].b.shape(), residual.shape())

                    input_to_layer = hidden_states
                else:
                    var inputs_to_tm = session.new_tensor_map()
                    results = xyz_graph.execute("input0", new_query_states, "input1", new_key_states, "input2", value_states)
                    var attn_output_t = results.get[DType.float32]("output0")

                    attn_output_r = attn_output_t.reshape((bsz, q_len, hidden_size))
                    
                    inputs_to_tm.borrow("input0", attn_output_r)
                    inputs_to_tm.borrow("input1", ln_out)
                    inputs_to_tm.borrow("input2", outproj_lin[i].w)
                    inputs_to_tm.borrow("input3", outproj_lin[i].b)
                    inputs_to_tm.borrow("input4", fc1_lin[i].w)
                    inputs_to_tm.borrow("input5", fc1_lin[i].b)
                    inputs_to_tm.borrow("input6", fc2_lin[i].w)
                    inputs_to_tm.borrow("input7", fc2_lin[i].b)
                    inputs_to_tm.borrow("input8", residual)

                    results = xyz2_graph.execute(inputs_to_tm)
                    hidden_states = results.get[DType.float32]("output0")
                    
                    print(attn_output_r.shape(), ln_out.shape(), outproj_lin[i].w.shape(), outproj_lin[i].b.shape(),
                          fc1_lin[i].w.shape(), fc1_lin[i].b.shape(), fc2_lin[i].w.shape(), fc2_lin[i].b.shape(), residual.shape())
                    
                    input_to_layer = hidden_states

            if words == 0:
                j_index = input_to_layer.shape()[1]
                new = Tensor[DType.float32] (1,1,hidden_size)
                for i in range(1):
                    for j in range(1):
                        for k in range(hidden_size):
                            new[Index(i,j,k)] = input_to_layer[Index(0,j_index-1,k)]
            else:
                new = input_to_layer
            
            var inputs_to_tm4 = session.new_tensor_map()
            inputs_to_tm4.borrow("input0", new)
            inputs_to_tm4.borrow("input1", lm_head_ln.w)
            inputs_to_tm4.borrow("input2", lm_head_ln.b)
            inputs_to_tm4.borrow("input3", lm_head_lin.w)
            inputs_to_tm4.borrow("input4", lm_head_lin.b)

            results = head_model.execute(inputs_to_tm4)
            lm_lin = results.get[DType.float32]("output0")

            print(new.shape(), lm_head_ln.w.shape(), lm_head_ln.b.shape(), lm_head_lin.w.shape(), lm_head_lin.b.shape())

            here = mypython.argmax_index(tensor_to_numpy(lm_lin))
            values.append(here[0][0])

            if here[0][0] == 50256:
                break
            input_to_layer = lm_lin

            if words == 0:
                new_end = now()
                first_token_time = (new_end - start)/1000000000
            words +=1

        end = now()
        total_time = (end - start)/1000000000
        print("total tokens:", len(values))
        print("Total time taken(llm): ",total_time)
        print("time per token: ", total_time/ len(values))
        print("first token time: ", first_token_time)
        print("Avg consecutive token time:", (total_time-first_token_time)/(len(values)-1))
        
        var np = Python.import_module("numpy")
        var np_values = np.zeros((1, len(values)), np.int32)
        for i in range(np_values.shape[0]):
            for j in range(np_values.shape[1]):
                np_values[i][j] = values[j]
        
        output = mypython.decode(np_values)
        print(output)

