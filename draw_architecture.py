import base64
import requests
import json
import os

mermaid_code = """
graph TD
    A[Raw Ethereum Transaction Sequence] --> B(Sliding Window Algorithm)
    B -->|W=200, S=50| C[Bag of Windows]
    C --> D1[BERT Embeddings 64-dim]
    C --> D2[Heuristics 4-dim]
    
    subgraph TMIL-ETH Architecture
        D1 --> E[Concat Feature Vector 68-dim]
        D2 --> E
        E -->|MLP| F[Instance Representations]
        F --> G[Gated Attention Mechanism]
        G -->|Attention Weights| H[Pooled Account Representation]
        H --> I[Phishing Probability]
        H --> J[Phish-Masked Contrastive Loss]
    end
    
    G -.->|High Attention| K[Forensic Localization Hit]
"""

def generate_mermaid_image():
    # Mermaid.ink expects base64 encoded diagram state
    # Create the state object
    state = {
        "code": mermaid_code,
        "mermaid": {"theme": "default"}
    }
    
    # Serialize to JSON and encode to Base64
    json_str = json.dumps(state)
    b64_str = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')
    
    url = f"https://mermaid.ink/img/{b64_str}"
    
    print(f"Fetching from: {url}")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("architecture.png", "wb") as f:
            f.write(response.content)
        print("Successfully saved architecture.png")
    else:
        print(f"Failed to fetch image. Status: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    generate_mermaid_image()
