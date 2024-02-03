from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
from transformers_stream_generator import init_stream_support
import warnings
warnings.filterwarnings("ignore", module="transformers")
logging.set_verbosity(logging.CRITICAL)

init_stream_support()

choice=input("choose a model\n1. Hermes SOLAR\n2. OpenChat GPTQ\n3. NatureAI 16k\n4. OpenChat16K\n")
modelname=[
    "TheBloke_Nous-Hermes-2-SOLAR-10.7B-GPTQ",
    "TheBloke_openchat-3.5-0106-GPTQ",
    "NurtureAI_openchat_3.5-16k-4bit-gptq",
    "TheBloke_openchat_3.5-16k-GPTQ"]
    
model_name_or_path =  "/home/sujit/Downloads/text-generation-webui-main/models/"+modelname[int(choice)-1]
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
context=""

while True:
    rawprompt=""
    while rawprompt=="":
        rawprompt=input("User: ")
    print("Assistant: ", end="")
    prompt=context+"User: "+rawprompt+"<|end_of_turn|>\nAssistant: "    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    generator =  model.generate(inputs=input_ids, temperature=0.01, do_stream=True, do_sample=True, top_p=0.01, top_k=2, max_new_tokens=14000, stream=True)
    output = words = "";last_tokens = [];firstline=False

    for index, x in enumerate(generator):
        tokens = x.cpu().tolist()
        word = tokenizer.decode(tokens)
        if "�" not in word:
            if firstline==True or (firstline==False and word!="\n"):    
                a=tokenizer.decode(last_tokens+tokens)
                if " " in a:
                    word = " " + word
                    last_tokens = []
                    firstline=True
                if "<" in a:
                    if ">" not in a:
                        last_tokens += tokens
                    else:
                        last_tokens=[]
                    if "<|end_of_turn|>" in a or "<|im_end|>" in a:
                        context=context+"User: "+rawprompt+"\nAssistant: "+output+"\n"
                        print();
                        #print("\n"+context)
                        break 
                else:
                    last_tokens = tokens
                    output = output+word
                    print(word,end="")

