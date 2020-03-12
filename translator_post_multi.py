import hug
import torch
import fairseq

# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]

# Load a transformer trained on WMT'16 En-De
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')
en2de.eval()  # disable dropout

en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')
en2fr.eval()  # disable dropout


en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru', tokenizer='moses', bpe='subword_nmt')
en2ru.eval()  # disable dropout



# The underlying model is available under the *models* attribute
#assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)



@hug.post(examples='txt="Hello World"')
@hug.local()
def translate(msg: hug.types.text, lang: hug.types.text):
	if lang == "fr":
		model = en2fr
	elif lang == "de":
		model = en2de
	elif lang == "ru"
		model == "en2ru"
	return {'msg':model.translate(msg)}
