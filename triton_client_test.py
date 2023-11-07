def py_requests():
    """
    通过python的requests库，直接访问Triton server，构造符合config.pbtxt的输入，以0为例
    """
    import requests
    import json

    url = 'http://10.58.253.38:10003/v2/models/bert/infer'  # HTTP访问端口是8000，可以在宿主机或者别的机器访问，注意IP

    headers = {'Content-Type': 'application/json'}
    data = {
        'inputs': [
            {
                'name': 'input_ids',
                'shape': [1, 128],
                'datatype': 'INT64',
                'data': [0] * 128
            },
            {
                'name': 'attention_mask',
                'shape': [1, 128],
                'datatype': 'INT64',
                'data': [0] * 128
            },
            {
                'name': 'token_type_ids',
                'shape': [1, 128],
                'datatype': 'INT64',
                'data': [0] * 128
            }
        ],
        'outputs': [
            {
                'name': 'output',
                'shape': [1, 768],
                'datatype': 'FP32'
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = json.loads(response.content.decode('utf-8'))

    print(result)  # 输出结果是一个768维度的模型


def pyt_client():
    """
    通过Triton client发送请求，以Hugging Face的tokenizer构建输入，发送请求，获取结果
    """
    import tritonclient.http as httpclient
    import numpy as np
    from transformers import BertTokenizer

    test_text = "人生全靠浪，我叫浪里个浪"
    model_path = "/home/work/var/data/ssr-share-data/llm/bert/bert-base-chinese/"

    tokenizer = BertTokenizer.from_pretrained(model_path)
    inputs = tokenizer(test_text, return_tensors="pt")
    
    triton_client = httpclient.InferenceServerClient(url='10.58.253.38:10003', verbose=False)  # 与 Triton 推理服务器的建立连接，固定不用改

    # 构建输入，要与config.pbtxt写的参数名称和数据类型保持一致
    inputs_val = [
        httpclient.InferInput("input_ids", inputs["input_ids"].shape, datatype="INT64"),       # 变量名、shape、type
        httpclient.InferInput("attention_mask", inputs["attention_mask"].shape, datatype="INT64"),  # 三个参数与写的config.pbtxt一致
        httpclient.InferInput("token_type_ids", inputs["token_type_ids"].shape, datatype="INT64")   # 这个地方仅是形成占位符，没有赋值
    ]
    
    # 对上述变量进行赋值
    val = [0] * 128
    inputs_val[0].set_data_from_numpy(inputs['input_ids'].numpy(), binary_data=True)
    inputs_val[1].set_data_from_numpy(inputs["attention_mask"].numpy(), binary_data=True)
    inputs_val[2].set_data_from_numpy(inputs["token_type_ids"].numpy(), binary_data=True)
    
    # # 根据config.pbtxt生成输出变量
    # outputs = [httpclient.InferRequestedOutput('output')]
    print("inputs:\n", inputs_val)
    # 根据config.pbtxt文件中name定义的模型名称，获取结果，此处为bert
    results = triton_client.infer(model_name="bert", inputs=inputs_val, outputs=None)
    results = results.as_numpy('output')  # 将client的对象类型转numpy
    print(len(results))


if __name__ == "__main__":
    pyt_client()
