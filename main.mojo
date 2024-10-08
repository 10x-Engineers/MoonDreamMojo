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

alias dim = 1152
alias num_heads = 16
alias head_dim = dim // num_heads
alias load_size = 256
alias load_size2 = 128
alias sequence_length = 729
alias pi_sqrt = 0.7978845608028654
alias total_VIT_blocks = 27


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

    fn forward(self, inout input:Tensor[DType.float32] , multiplication:Model, addition:Model, transpose:Model, 
               transpose_5:Model) raises -> Tensor[DType.float32]:
        var b:Int = input.shape()[0]
        var c:Int = input.shape()[1]
        var hp1:Int = input.shape()[2]
        var wp2:Int = input.shape()[3]
        var p1:Int = 14
        var p2:Int = 14
        h, w = hp1 // p1, wp2 // p2

        var t:TensorShape = (b, c, h, p1, w, p2)
        var x = input.reshape(t)

        var results = transpose_5.execute("input0", x)
        var permute = results.get[DType.float32]("output0")
        t = (b, h * w, c * p1 * p2)
        x = permute.reshape(t)
        results = transpose.execute("input0", self.weights)
        var W_T = results.get[DType.float32]("output0")
        results = multiplication.execute("input0", x, "input1", W_T)
        var output = results.get[DType.float32]("output0")
        results = addition.execute("input0", output, "input1", self.biases)
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

    fn unbind_qkv(self, qkv: Tensor[DType.float32], inout q: Tensor[DType.float32], inout start_qkv: Int):
        var start_q = 0
        var q_num_elements = q.num_elements()
        for i in range(0, q_num_elements, load_size):
            q.store(start_q, qkv.load[width=load_size](start_qkv))
            start_q += load_size
            start_qkv += load_size

    fn scaled_dot_product_attention(self, query:Tensor[DType.float32], key:Tensor[DType.float32], value:Tensor[DType.float32],
                                    transpose_21:Model, multiplication_4D:Model, softmax:Model, multiplication_4D_2:Model) 
                                    raises -> Tensor[DType.float32]:

        var base = Float32(query.shape()[-1])
        var exponent = Float32(0.5)
        var scale_factor:Float32 = 1 / pow(base, exponent)

        var results = transpose_21.execute("input0", key)
        var key_transpose = results.get[DType.float32]("output0")
        results = multiplication_4D.execute("input0", query, "input1", key_transpose)
        var atten_weights = results.get[DType.float32]("output0")
        var attention_weights = atten_weights * scale_factor


        var attn_weights = Tensor[DType.float32] (attention_weights.shape())
        for i in range(attn_weights.shape()[0]):
            for j in range(attn_weights.shape()[1]):
                for k in range(attn_weights.shape()[2]):
                    var new_tens = Tensor[DType.float32] (attn_weights.shape()[3])
                    new_tens.store(0, attention_weights.load[width = sequence_length](i,j,k,0))
                    var results = softmax.execute("input0", new_tens)
                    var xd = results.get[DType.float32] ("output0")
                    attn_weights.store(Index(i,j,k,0), xd.load[width = sequence_length](0))

        results = multiplication_4D_2.execute("input0", attn_weights, "input1", value)
        var output = results.get[DType.float32]("output0")
        return output


    fn forward(self, inout input:Tensor[DType.float32] , multiplication:Model, addition:Model, transpose:Model, 
               transpose_4:Model, transpose_21:Model, multiplication_4D:Model, softmax:Model, multiplication_4D_2:Model,
               transpose_12:Model, multiplication_32:Model) raises -> Tensor[DType.float32]:
        var B = input.shape()[0]
        var N = input.shape()[1]
        var C = input.shape()[2]

        var results = transpose.execute("input0", self.weights)
        var W_T = results.get[DType.float32]("output0")
        results = multiplication.execute("input0", input, "input1", W_T)
        var output = results.get[DType.float32]("output0")
        results = addition.execute("input0", output, "input1", self.biases)
        output = results.get[DType.float32]("output0")
        
        var t:TensorShape = (B, N, 3, num_heads, head_dim)
        var rs = output.reshape(t)

        results = transpose_4.execute("input0", rs)
        var qkv = results.get[DType.float32]("output0")

        var t1:TensorShape = (qkv.shape()[1],qkv.shape()[2],qkv.shape()[3],qkv.shape()[4])
        var q = Tensor[DType.float32] (t1)
        var k = Tensor[DType.float32] (t1)
        var v = Tensor[DType.float32] (t1)
        var start_qkv_q = 0
        var start_qkv_k = q.num_elements()
        var start_qkv_v = start_qkv_k + k.num_elements()
        self.unbind_qkv(qkv, q, start_qkv_q)
        self.unbind_qkv(qkv, k, start_qkv_k)
        self.unbind_qkv(qkv, v, start_qkv_v) 
        var x = self.scaled_dot_product_attention(q,k,v,transpose_21, multiplication_4D, softmax, multiplication_4D_2)
        
        results = transpose_12.execute("input0", x)
        var x_t = results.get[DType.float32]("output0")
        t = (B,N,C)
        var x_r = x_t.reshape(t)

        results = transpose.execute("input0", self.proj_w)
        var PW_T = results.get[DType.float32]("output0")
        results = multiplication_32.execute("input0", x_r, "input1", PW_T)
        var atten_out = results.get[DType.float32]("output0")
        results = addition.execute("input0", atten_out, "input1", self.proj_b)
        output = results.get[DType.float32]("output0")

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
    var graph0 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d", "e", "f")))
    var t1 = ops.transpose(graph0[0], 1, 2)
    var t2 = ops.transpose(t1, 3, 4)
    var t3 = ops.transpose(t2, 2, 3)
    graph0.output(t3)
    graph0.verify()
    var session = engine.InferenceSession()
    var transpose_5 = session.load(graph0)
    print(".", end = " ")

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

    var graph4 = Graph(in_types=List[Type](TensorType(DType.float32, 1, 729, 1152),TensorType(DType.float32, 1152), 
                                           TensorType(DType.float32, 1152)))
    var mean = ops.layer_norm(graph4[0],gamma = graph4[1], beta = graph4[2] , epsilon = 1e-5)
    graph4.output(mean)
    graph4.verify()
    var norm = session.load(graph4)
    print(".", end = " ")

    var graph5 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d", "e")))
    var t4 = ops.transpose(graph5[0], 1, 2)
    var t5 = ops.transpose(t4, 0, 1)
    var t6 = ops.transpose(t5, 2, 3)
    graph5.output(t6)
    graph5.verify()
    var transpose_4 = session.load(graph5)
    print(".", end = " ")

    var graph6 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d")))
    transposed = ops.transpose(graph6[0],-2,-1)
    graph6.output(transposed)
    graph6.verify()
    var transpose_21 = session.load(graph6)
    print(".", end = " ")

    var graph7 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d"), 
                                           TensorType(DType.float32, "a", "b", "d", "c")))
    var out7 = graph7[0] @ graph7[1]
    graph7.output(out7)
    graph7.verify()
    var multiplication_4D = session.load(graph7)
    print(".", end = " ")

    var graph8 = Graph(in_types=List[Type](TensorType(DType.float32, "a")))
    var softmaxed = ops.softmax(graph8[0])
    graph8.output(softmaxed)
    graph8.verify()
    var softmax = session.load(graph8)
    print(".", end = " ")

    var graph9 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "m", "n", "n"), 
                                           TensorType(DType.float32, "a", "m", "n", "x")))
    var out9 = graph9[0] @ graph9[1]
    graph9.output(out9)
    graph9.verify()
    var multiplication_4D_2 = session.load(graph9)
    print(".", end = " ")

    var graph10 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d")))
    transposed = ops.transpose(graph10[0],1,2)
    graph10.output(transposed)
    graph10.verify()
    var transpose_12 = session.load(graph10)
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

    var graph13 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b")))
    transposed = ops.transpose(graph13[0], 0, 1)
    graph13.output(transposed)
    graph13.verify()
    var transpose_01 = session.load(graph13)
    print(".", end = " ")

    var graph14 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c")))
    transposed = ops.transpose(graph14[0],1,2)
    graph14.output(transposed)
    graph14.verify()
    var transpose3_12 = session.load(graph14)
    print(".", end = " ")

    var in_types = List[Type] (TensorType(DType.float32, 1, 729, 1152), TensorType(DType.float32, 1, 729, 1152))
    var graph15 = Graph(in_types=in_types)
    var inputs = List[Symbol] (graph15[0], graph15[1])
    var c = ops.concat(inputs, 2)
    graph15.output(c)
    graph15.verify()
    var concat = session.load(graph15)
    


    ###################################################################################################################
    print()

    var weights = load("weights.maxckpt")

    var mypython = Python.import_module("helper")
    py_builtins = Python.import_module("builtins")
    var image_path = py_builtins.input("Enter image path: ")
    print(image_path)
    print("Compiling Model", end = " ")
    # var image_path = "flower.jpeg"
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
    var FC_List_1 = List[FC] ()
    var FC_List_2 = List[FC] ()

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
        FC_List_1.append(FC(fc_1_weight, fc_1_bias))

        fc_2_weight = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.mlp.fc2.weight')
        fc_2_bias = weights.get[DType.float32]('encoder.model.visual.blocks.'+str(i)+'.mlp.fc2.bias')
        FC_List_2.append(FC(fc_2_weight, fc_2_bias))
        print(".", end = " ")

    last_layer_norm_weight = weights.get[DType.float32]('encoder.model.visual.norm.weight')
    last_layer_norm_bias = weights.get[DType.float32]('encoder.model.visual.norm.bias')
    var last_layer_norm = LayerNorm(last_layer_norm_weight, last_layer_norm_bias)
    print(".", end = " ")
    #*#*#*#*#*#*#*#*

    last_fc1_weight = weights.get[DType.float32]('projection.mlp.fc1.weight')
    last_fc1_bias = weights.get[DType.float32]('projection.mlp.fc1.bias')
    var last_fc1 = FC(last_fc1_weight, last_fc1_bias)

    last_fc2_weight = weights.get[DType.float32]('projection.mlp.fc2.weight')
    last_fc2_bias = weights.get[DType.float32]('projection.mlp.fc2.bias')
    var last_fc2 = FC(last_fc2_weight, last_fc2_bias)

    print()
    print("Running model")

    var start = now()

    var patch_embed = patch_embedding.forward(preprocessed_image,multiplication, addition, transpose, transpose_5)
    var pos_embed = patch_embed + positional_embedding

    var x = pos_embed

    for i in range(total_VIT_blocks):
        var ln = Layer_Norm_1_List[i].forward(x, norm)
        var attention = Attention_List[i].forward(ln, multiplication, addition, transpose, transpose_4, transpose_21, 
                                                  multiplication_4D, softmax, multiplication_4D_2, transpose_12, 
                                                  multiplication_32)
        var attention_out = attention + x
        var ln2 = Layer_Norm_2_List[i].forward(attention_out, norm)
        var fc1_out = FC_List_1[i].forward(ln2,transpose,multiplication_32,addition)
        var g = Gelu(fc1_out,tanh)
        var fc2_out = FC_List_2[i].forward(g,transpose,multiplication_32,addition)
        var mlp_out = fc2_out + attention_out
        x = mlp_out

    var full_img_features = last_layer_norm.forward(x,norm)

    var s:TensorShape = (729,1152)
    var full_img_features_reshaped = Tensor[DType.float32](s)
    var q_num_elements = full_img_features_reshaped.num_elements()
    var start_q = 0
    for i in range(0, q_num_elements, load_size2):
        full_img_features_reshaped.store(start_q, full_img_features.load[width=load_size2](start_q))
        start_q += load_size2

    var results = transpose_01.execute("input0", full_img_features_reshaped)
    var t = results.get[DType.float32]("output0")

    s = (1152,27,27)
    var r = t.reshape(s)

    s = (1, 1152, 729)
    var new_reshaped_patch_features = r.reshape(s)
    results = transpose3_12.execute("input0", new_reshaped_patch_features)
    var reshaped_patch_features_final = results.get[DType.float32]("output0")

    results = concat.execute("input0", full_img_features, "input1", reshaped_patch_features_final)
    var final_features = results.get[DType.float32] ("output0")

    var lfc1 = last_fc1.forward(final_features, transpose, multiplication_32, addition)
    var lg = Gelu(lfc1,tanh)
    var lfc2 = last_fc2.forward(lg,transpose, multiplication_32,addition)

    print("lfc2:\n", lfc2)
    tensors = TensorDict()
    tensors.set("x", lfc2)

    save(tensors,"encoder_output.maxckpt")

    var end = now()
    print("total generation time: ",(end - start)/1000000000)


    