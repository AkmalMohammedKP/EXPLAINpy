from flask import jsonify,request
from config import app
import pydoc
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

import logging
# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# Load pre-trained model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure tokenizer.pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Initialize an empty list to store feedback data
feedback_data = []
def preprocess_and_train_bot(feedback_data):
    # Define lists to store preprocessed data
    input_texts = []
    target_texts = []

    # Preprocess feedback data
    for feedback in feedback_data:
        # Tokenize and encode original code and user-explained code
        original_code_tokens = tokenizer.encode(feedback["original_code"], return_tensors="pt", padding=True, truncation=True)
        user_explanation_tokens = tokenizer.encode(feedback["user_explanation"], return_tensors="pt", padding=True, truncation=True)
        
        # Combine original code and user-explained code as input
        input_text = torch.cat((original_code_tokens, user_explanation_tokens), dim=1)
        
        # Tokenize and encode bot-generated explanation as target
        target_text = tokenizer.encode(feedback["bot_explanation"], return_tensors="pt", padding=True, truncation=True)
        
        # Append preprocessed data to lists
        input_texts.append(input_text)
        target_texts.append(target_text)

    # Concatenate lists to create tensors
    input_texts = torch.cat(input_texts, dim=0)
    target_texts = torch.cat(target_texts, dim=0)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        for input_text, target_text in zip(input_texts, target_texts):
            # Forward pass
            outputs = model(input_text)
            logits = outputs.logits  # Extract logits from the outputs

            # Reshape logits and target_texts for the loss calculation
            logits = logits.view(-1, logits.shape[-1])

            # Ensure that both tensors have the same batch size
            if logits.size(0) != target_text.size(0):
                # Resize target_text to match logits's batch size
                target_text = target_text[:logits.size(0)]

            # Calculate the loss
            loss = criterion(logits, target_text)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    
    # Evaluate the model (optional)
    # Implement evaluation logic here

    # Save the trained model (optional)
    # Implement model saving logic here


@app.route('/module/<string:name>',methods=["GET"])
def search_module(name):
    try:
        module_doc_text=pydoc.render_doc(name)
    except Exception as e:
        return jsonify({"moduleName":name,"documentation":"Module not found."}),404
    return jsonify({"moduleName":name,
                    "documentaion":module_doc_text
                    }),200


@app.route('/code',methods=["POST"])
def explain_code():
    """{
        "code":code
    }
    """
    code=request.json.get("code")

    # Tokenize input sequence and set attention mask
    input_ids = tokenizer.encode(code, return_tensors="pt", padding=True, truncation=True)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    # Run inference
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode the generated output
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if code in explanation and len(explanation)<len(code)+8:
        return jsonify({"code": code, "explanation": "Error."}), 400
    else:
        return jsonify({"code": code, "explanation": explanation}), 200


@app.route('/feedback',methods=["POST"])
def handle_feedback():
    """{
        "originalCode":code,
        "botExplanation":....,
        "userExplanation":....
    }"""
    original_code=request.json.get("originalCode")
    bot_explanation=request.json.get("botExplanation")
    user_explanation=request.json.get("userExplanation")

    feedback_data.append({"original_code": original_code, "bot_explanation": bot_explanation, "user_explanation": user_explanation})

    preprocess_and_train_bot(feedback_data)
    
    return jsonify({"message": "Thank you for your feedback!"}), 200



if __name__=="__main__":
    app.run(debug=True)


