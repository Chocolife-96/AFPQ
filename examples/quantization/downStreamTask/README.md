## code

cd code

bash gen_preds.sh /root/model/models/WizardCoder-Python-7B ./preds/7b/n2f3-g64/ ../llm-awq/awq_cache/wizardcode/7b-n2f3-g64.pt 3 n2f3 64

## gsm8k

cd gsm8k/test

bash test.sh /root/model/models/MetaMath-7B-V1.0/ ../json_out/7B/n2f3-g64/