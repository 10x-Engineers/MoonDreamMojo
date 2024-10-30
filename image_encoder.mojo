from max.engine import InputSpec, InferenceSession
from python import Python, PythonObject
from utils.index import Index
from time import now
from max.graph import Graph, ops, Symbol
from max.graph.type import TensorType, Dim, Type
from max import engine
from max.tensor import Tensor, TensorShape
from max.engine import Model
from algorithm import sum
from utils.numerics import inf
from algorithm import parallelize
from memory import memcpy, memcmp, memset_zero
from max.graph.checkpoint import save, TensorDict, load

alias batch_size = 1
alias sequence_length = 729
alias dim = 1152
alias num_heads = 16
alias head_dim = dim // num_heads
alias total_VIT_blocks = 27
alias pi_sqrt = 0.7978845608028654
alias scale_factor = 0.11785112321376801

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

struct PatchEmbedding:
    var weights: Tensor[DType.float32]
    var biases: Tensor[DType.float32]

    fn __init__(inout self, W: Tensor[DType.float32], B: Tensor[DType.float32]):
        self.weights = W
        self.biases = B

    fn forward(self, inout input:Tensor[DType.float32] , model:Model) raises -> Tensor[DType.float32]:
        results = model.execute("input0", input, "input1", self.weights, "input2", self.biases)
        output = results.get[DType.float32]("output0")
        return output

struct LayerNorm(CollectionElement):
    var gamma: Tensor[DType.float32]
    var beta: Tensor[DType.float32]

    fn __init__(inout self, gema: Tensor[DType.float32], beta: Tensor[DType.float32]):
        self.gamma = gema
        self.beta = beta

    fn __copyinit__(inout self, existing: Self):
        self.gamma = existing.gamma
        self.beta = existing.beta
    
    fn __moveinit__(inout self, owned existing: Self):
        self.gamma = existing.gamma^
        self.beta = existing.beta^
    
    fn forward(self, input: Tensor[DType.float32], norm: Model) raises -> Tensor[DType.float32]:
        var results = norm.execute("input0", input, "input1", self.gamma, "input2", self.beta)
        var output = results.get[DType.float32]("output0")
        return output

struct Attention:
    var weights: Tensor[DType.float32]
    var biases: Tensor[DType.float32]
    var proj_w: Tensor[DType.float32]
    var proj_b: Tensor[DType.float32]

    fn __init__(inout self, W: Tensor[DType.float32], B: Tensor[DType.float32], P_W: Tensor[DType.float32], 
                P_B: Tensor[DType.float32]):
        self.weights = W
        self.biases = B
        self.proj_w = P_W
        self.proj_b = P_B

    fn __copyinit__(inout self, existing: Self):
        self.weights = existing.weights
        self.biases = existing.biases
        self.proj_w = existing.proj_w
        self.proj_b = existing.proj_b
    
    fn __moveinit__(inout self, owned existing: Self):
        self.weights = existing.weights^
        self.biases = existing.biases^
        self.proj_w = existing.proj_w^
        self.proj_b = existing.proj_b^


    fn forward(self, inout wow:Tensor[DType.float32] , model_start:Model, scaled_dot_product_attention_graph:Model, 
               model_end:Model, qkv_:Model) raises -> Tensor[DType.float32]:
        results = model_start.execute("input0", wow, "input1", self.weights, "input2", self.biases)
        var qkv = results.get[DType.float32]("output0")
        results = qkv_.execute("input0", qkv)
        var q = results.get[DType.float32]("output0")
        var k = results.get[DType.float32]("output1")
        var v = results.get[DType.float32]("output2")
        results = scaled_dot_product_attention_graph.execute("input0", q, "input1", k, "input2", v)
        var x = results.get[DType.float32]("output0")
        results = model_end.execute("input0", x, "input1", self.proj_w, "input2", self.proj_b)
        var output = results.get[DType.float32]("output0")
        return output

struct FC(CollectionElement):
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]

    fn __init__(inout self, w: Tensor[DType.float32], b: Tensor[DType.float32]):
        self.weight = w
        self.bias = b

    fn __copyinit__(inout self, existing: Self):
        self.weight = existing.weight
        self.bias = existing.bias
    
    fn __moveinit__(inout self, owned existing: Self):
        self.weight = existing.weight^
        self.bias = existing.bias^
    
    fn forward(self, input: Tensor[DType.float32], transpose:Model, multiplication_32: Model, addition: Model) 
              raises -> Tensor[DType.float32]:
        var results = transpose.execute("input0", self.weight)
        var W_T = results.get[DType.float32]("output0")
        results = multiplication_32.execute("input0", input, "input1", W_T)
        var mul_out = results.get[DType.float32]("output0")
        results = addition.execute("input0", mul_out, "input1", self.bias)
        output = results.get[DType.float32]("output0")
        return output

struct FCNew(CollectionElement):
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]

    fn __init__(inout self, w: Tensor[DType.float32], b: Tensor[DType.float32]):
        self.weight = w
        self.bias = b

    fn __copyinit__(inout self, existing: Self):
        self.weight = existing.weight
        self.bias = existing.bias
    
    fn __moveinit__(inout self, owned existing: Self):
        self.weight = existing.weight^
        self.bias = existing.bias^
    
    fn forward(self, input: Tensor[DType.float32], model:Model) raises -> Tensor[DType.float32]:
        results = model.execute("input0", input, "input1", self.weight, "input2", self.bias)
        output = results.get[DType.float32]("output0")
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
    print(".", end = " ")

    var graph4 = Graph(in_types=List[Type](TensorType(DType.float32, batch_size, sequence_length, dim),
                                           TensorType(DType.float32, dim), TensorType(DType.float32, dim)))
    var mean = ops.layer_norm(graph4[0],gamma = graph4[1], beta = graph4[2] , epsilon = 1e-5)
    graph4.output(mean)
    graph4.verify()
    var norm = session.load(graph4)
    print(".", end = " ")

    var graph11 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "n","x")))
    var out11 = graph11[0] @ graph11[1]
    graph11.output(out11)
    graph11.verify()
    var multiplication_32 = session.load(graph11)
    print(".", end = " ")

    var graph12 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c")))
    var tanhed = ops.tanh(graph12[0])
    graph12.output(tanhed)
    graph12.verify()
    var tanh = session.load(graph12)
    print(".", end = " ")

    var graph_preprocessing = Graph(in_types=List[Type](TensorType(DType.float32, batch_size, 3, 378, 378), 
                              TensorType(DType.float32, dim, 588), TensorType(DType.float32, dim)))
    var p1:Int = 14
    var p2:Int = 14
    h, w = 378 // p1, 378 // p2

    var reshape = graph_preprocessing[0].reshape(1,3,h,p1,w,p2)
    var tt1 = ops.transpose(reshape, 1, 2)
    var tt2 = ops.transpose(tt1, 3, 4)
    var tt3 = ops.transpose(tt2, 2, 3)
    var permute = tt3.reshape(batch_size, h * w, 3 * p1 * p2)
    var ttransposed = ops.transpose(graph_preprocessing[1],-1,-2)
    var mult = permute @ ttransposed
    var add = mult + graph_preprocessing[2]
    graph_preprocessing.output(add)
    graph_preprocessing.verify()
    var tpatch_embedding = session.load(graph_preprocessing)
    print(".", end = " ")

    var graph_attention = Graph(in_types=List[Type](TensorType(DType.float32, batch_size, sequence_length, dim), 
                                TensorType(DType.float32, 3456, dim), TensorType(DType.float32, 3456)))
    
    var transposedt = ops.transpose(graph_attention[1],-1,-2)
    var multt = graph_attention[0] @ transposedt
    var addt = multt + graph_attention[2]
    var reshaped = addt.reshape(batch_size, sequence_length, 3, num_heads, head_dim)
    var tt4 = ops.transpose(reshaped, batch_size, 2)
    var tt5 = ops.transpose(tt4, 0, 1)
    var qkv = ops.transpose(tt5, 2, 3)
    graph_attention.output(qkv)
    graph_attention.verify()
    var attnetion_start = session.load(graph_attention)
    print(".", end = " ")

    var dot_product_attention_graph = Graph(in_types=List[Type](
                                      TensorType(DType.float32, batch_size, num_heads, sequence_length, head_dim), 
                                      TensorType(DType.float32, batch_size, num_heads, sequence_length, head_dim), 
                                      TensorType(DType.float32, batch_size, num_heads, sequence_length, head_dim)))
    var transposedtt = ops.transpose(dot_product_attention_graph[1],-2,-1)
    var m = dot_product_attention_graph[0] @ transposedtt
    var sf = m * scale_factor
    var softmx = ops.softmax(sf)
    var ans = softmx @ dot_product_attention_graph[2]
    dot_product_attention_graph.output(ans)
    dot_product_attention_graph.verify()
    var scaled_dot_product_attention_graph = session.load(dot_product_attention_graph)
    print(".", end = " ")

    var graph_attention_final = Graph(in_types=List[Type](
                                TensorType(DType.float32, batch_size, num_heads, sequence_length, head_dim), 
                                TensorType(DType.float32, dim, dim), TensorType(DType.float32, dim)))
    var tttransposed = ops.transpose(graph_attention_final[0],1,2)
    var x_r = tttransposed.reshape(batch_size, sequence_length, dim)
    var PW_T = ops.transpose(graph_attention_final[1],-1,-2)
    var m1 = x_r @ PW_T
    var out_final = m1 + graph_attention_final[2]
    graph_attention_final.output(out_final)
    graph_attention_final.verify()
    var attention_end = session.load(graph_attention_final)
    print(".", end = " ")

    var gelu = Graph(in_types=List[Type](TensorType(DType.float32, batch_size, sequence_length, 4304)))
    var p = gelu[0] * gelu[0] * gelu[0]
    var a = 0.044715 * p
    var m2 = gelu[0]+a
    var m3 = pi_sqrt * m2
    var tanh_out = ops.tanh(m3)
    plus = 1 + tanh_out
    result = 0.5*gelu[0]*plus
    gelu.output(result)
    gelu.verify()
    var gelu_graph = session.load(gelu)
    print(".", end = " ")

    var fc_1 = Graph(in_types=List[Type](TensorType(DType.float32, batch_size, sequence_length, dim), 
                     TensorType(DType.float32, 4304, dim), TensorType(DType.float32, 4304)))
    var w_t = ops.transpose(fc_1[1],-1,-2)
    var mu = fc_1[0] @ w_t
    var addi = mu + fc_1[2]
    fc_1.output(addi)
    fc_1.verify()
    var fc1_model = session.load(fc_1)
    print(".", end = " ")

    var fc_2 = Graph(in_types=List[Type](TensorType(DType.float32, batch_size, sequence_length, 4304), 
                     TensorType(DType.float32, dim, 4304), TensorType(DType.float32, dim)))
    w_t = ops.transpose(fc_2[1],-1,-2)
    mu = fc_2[0] @ w_t
    addi = mu + fc_2[2]
    fc_2.output(addi)
    fc_2.verify()
    var fc2_model = session.load(fc_2)
    print(".", end = " ")

    var post_model = Graph(in_types=List[Type](TensorType(DType.float32, batch_size, sequence_length, dim)))
    var rshp = post_model[0].reshape(sequence_length,dim)
    var trns =ops.transpose(rshp, 0, 1)
    var rshp1 = trns.reshape(dim,27,27)
    var rshp2 = rshp1.reshape(batch_size, dim, sequence_length)
    var trns1 = ops.transpose(rshp2,1,2)
    var inputss = List[Symbol] (post_model[0], trns1)
    var cc = ops.concat(inputss, 2)
    post_model.output(cc)
    post_model.verify()
    var post_modell = session.load(post_model)
    print(".", end = " ")

    var qkv_model = Graph(in_types=List[Type](TensorType(DType.float32, 3, batch_size, num_heads, sequence_length, head_dim)))
    var q = qkv_model[0][0:1, 0:1, 0:num_heads, 0:sequence_length, 0:head_dim].reshape(
                                       batch_size, num_heads, sequence_length,head_dim)
    var k = qkv_model[0][1:2, 0:1, 0:num_heads, 0:sequence_length, 0:head_dim].reshape(
                                       batch_size, num_heads, sequence_length,head_dim)
    var v = qkv_model[0][2:3, 0:1, 0:num_heads, 0:sequence_length, 0:head_dim].reshape(
                                       batch_size, num_heads, sequence_length,head_dim)
    var o = List[Symbol] ()
    o.append(q)
    o.append(k)
    o.append(v)
    qkv_model.output(o)
    qkv_model.verify()
    var qkv_ = session.load(qkv_model)
    print(".", end = " ")
    ###################################################################################################################
    print()
    var weights = load("weights.maxckpt")
    var mypython = Python.import_module("helper")
    py_builtins = Python.import_module("builtins")
    var image_path = py_builtins.input("Enter image path: ")

    print("Compiling Model", end = " ")
    var preprocessed_image:Tensor[DType.float32] = numpy_to_tensor( mypython.image_preprocessing(image_path))
    print(".", end = " ")

    patch_embed_weight = weights.get[DType.float32]("encoder.model.visual.patch_embed.linear.weight")
    patch_embed_bias = weights.get[DType.float32]("encoder.model.visual.patch_embed.linear.bias")
    var patch_embedding = PatchEmbedding(patch_embed_weight, patch_embed_bias)
    print(".", end = " ")

    positional_embedding = weights.get[DType.float32]("encoder.model.visual.pos_embed")
    print(".", end = " ")

    ##### VIT BLOCK #####
    var Layer_Norm_1_List = List[LayerNorm] ()
    var Attention_List = List[Attention] ()
    var Layer_Norm_2_List = List[LayerNorm] ()
    var FC_List_1 = List[FCNew] ()
    var FC_List_2 = List[FCNew] ()

    for i in range(total_VIT_blocks):
        layer_norm_weight = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.norm1.weight')
        layer_norm_bias = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.norm1.bias')
        Layer_Norm_1_List.append(LayerNorm(layer_norm_weight, layer_norm_bias))

        qkv_weight = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.attn.qkv.weight')
        qkv_bias = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.attn.qkv.bias')
        proj_weight = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.attn.proj.weight')
        proj_bias = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.attn.proj.bias')
        Attention_List.append(Attention(qkv_weight, qkv_bias, proj_weight, proj_bias))

        layer_norm_2_weight = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.norm2.weight')
        layer_norm_2_bias = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.norm2.bias')
        Layer_Norm_2_List.append(LayerNorm(layer_norm_2_weight, layer_norm_2_bias))

        fc_1_weight = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.mlp.fc1.weight')
        fc_1_bias = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.mlp.fc1.bias')
        FC_List_1.append(FCNew(fc_1_weight, fc_1_bias))

        fc_2_weight = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.mlp.fc2.weight')
        fc_2_bias = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.mlp.fc2.bias')
        FC_List_2.append(FCNew(fc_2_weight, fc_2_bias))
        print(".", end = " ")

    last_layer_norm_weight = weights.get[DType.float32]('encoder.model.visual.norm.weight')
    last_layer_norm_bias = weights.get[DType.float32]('encoder.model.visual.norm.bias')
    var last_layer_norm = LayerNorm(last_layer_norm_weight, last_layer_norm_bias)
    print(".", end = " ")

    last_fc1_weight = weights.get[DType.float32]('projection.mlp.fc1.weight')
    last_fc1_bias = weights.get[DType.float32]('projection.mlp.fc1.bias')
    var last_fc1 = FC(last_fc1_weight, last_fc1_bias)

    last_fc2_weight = weights.get[DType.float32]('projection.mlp.fc2.weight')
    last_fc2_bias = weights.get[DType.float32]('projection.mlp.fc2.bias')
    var last_fc2 = FC(last_fc2_weight, last_fc2_bias)

    print()
    print("Running model")

    var start = now()

    var patch_embed = patch_embedding.forward(preprocessed_image, tpatch_embedding)
    var pos_embed = patch_embed + positional_embedding
    var x = pos_embed

    for i in range(total_VIT_blocks):
        var ln = Layer_Norm_1_List[i].forward(x, norm)
        var attention = Attention_List[i].forward(ln, attnetion_start, scaled_dot_product_attention_graph, attention_end, qkv_)
        var attention_out = attention + x
        var ln2 = Layer_Norm_2_List[i].forward(attention_out, norm)
        var fc1_out = FC_List_1[i].forward(ln2,fc1_model)
        var result = gelu_graph.execute("input0", fc1_out)
        var g = result.get[DType.float32]("output0")
        var fc2_out = FC_List_2[i].forward(g,fc2_model)
        var mlp_out = fc2_out + attention_out
        x = mlp_out

    var full_img_features = last_layer_norm.forward(x,norm)
    results = post_modell.execute("input0", full_img_features)
    var final_features = results.get[DType.float32] ("output0")
    var lfc1 = last_fc1.forward(final_features, transpose, multiplication_32, addition)
    var lg = Gelu(lfc1,tanh)
    var lfc2 = last_fc2.forward(lg,transpose, multiplication_32,addition)
    
    var end = now()
    tensors = TensorDict()
    tensors.set("x", lfc2)

    save(tensors,"encoder_output.maxckpt")

    print("total image encoder time: ",(end - start)/1000000000)