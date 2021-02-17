# Neural Translator
Using pytorch pretrained fairseq 

## Available language model:
- French as fe
- Germen as de

## POST request
Please follow format:
> http://0a4e6a7e.ngrok.io/translate?msg=input_string&lang=target_langauge


# Accesing translate service 

```bash
# Start translate service using hug
hug -f translate.py

# Expose port via ngrok
brew install ngrok
ngrok http 8000

# web interface
localhost:8000/translate/{msg}

# curl 
curl localhost:8000/translate/{msg}
```
