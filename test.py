from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(
#https://github.com/microsoft/LLMLingua/issues/79
    # model_name="meta-llama/Llama-3.2-1B",
    # model_name="lgaalves/gpt2-dolly",

#https://github.com/microsoft/LLMLingua/issues/75
    #device_map="mps",
    device_map="cpu",
)
prompt = "Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He rearranged five of the boxes into packages of six highlighters each and sold them for $3 per package. He sold the rest of the highlighters separately at the rate of three pens for $2. How much did he make in total?"

compressed_prompt = llm_lingua.compress_prompt(prompt, instruction="", question="", target_token=200)
print(compressed_prompt)
