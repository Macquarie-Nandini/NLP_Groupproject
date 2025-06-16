import pandas as pd
import os
import logging
import re
from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, login
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import pytesseract

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Configuration
HF_TOKEN = "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"
MODEL_REPO = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_FILENAME = "llama-2-7b-chat.Q4_K_M.gguf"

# Initialize model
llm = None

def download_model():
    """Download model if not already present"""
    if not os.path.exists(MODEL_FILENAME):
        logger.info("Authenticating with Hugging Face...")
        login(token=HF_TOKEN)
        logger.info("Downloading model...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            token=HF_TOKEN,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        logger.info(f"Model downloaded to: {model_path}")

def load_model():
    """Load the LLaMA model"""
    global llm
    if not llm:
        download_model()
        logger.info("Loading model...")
        llm = Llama(
            model_path=MODEL_FILENAME,
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=0,
            verbose=True
        )
        logger.info("Model loaded successfully!")

# ===== Marketing Content Generation =====
def generate_marketing_content(data):
    """Generate structured marketing content with detailed prompt"""
    detailed_prompt = f"""**Marketing Content Generation Task**

**Output Format Requirements:**
Your response MUST use EXACTLY these section headers (including colons):
1. Company Name: 
2. Description: 
3. Industry Type: 
4. Tagline: 
5. Marketing Copy: 

**Business Details:**
- Core Business: {data['company_desc']}
- Location: {data['contact_details']}
- Contact: {data['email']} | {data['phone']}
- Founder: {data['founder_name']}
- Desired Tone: {data['tone']}

**Required Output Structure:**

1. **Company Name**: 
   <creative name reflecting business essence>

2. **Description**: 
   <300-word professional description covering:
   - Core offerings
   - Unique value proposition
   - Target audience
   - Founder's vision
   - Tone: {data['tone']}>

3. **Industry Type**: 
   <specific industry classification>

4. **Tagline**: 
   <catchy, memorable phrase under 10 words>

5. **Marketing Copy**: 
   <3-5 bullet points highlighting key benefits>

**Rules:**
- Maintain {data['tone']} tone consistently
- Be specific and avoid generic phrases
- Ensure all sections are completed
- Use clear section headers exactly as shown above
- No placeholders - provide complete content"""

    try:
        logger.debug("Generating marketing content with prompt...")
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": detailed_prompt}],
            max_tokens=1500,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )
        content = response['choices'][0]['message']['content']
        logger.debug(f"Raw LLM response: {content}")
        return content
    except Exception as e:
        logger.error(f"Marketing generation error: {str(e)}")
        return f"Error: {str(e)}"

# ===== Education Content Generation =====
def generate_education_content(data):
    """Generate structured educational content with detailed prompt"""
    detailed_prompt = f"""**Education Content Generation Task**

**Output Format Requirements:**
Your response MUST use EXACTLY these section headers (including colons):
1. Concept Summary: 
2. Quiz Questions: 
3. Real-life Examples: 
4. Key Analogies: 
5. Study Tips: 

**Education Details:**
- Subject: {data['subject']}
- Topic: {data['topic']}
- Level: {data['level']}
- Output Types: {data['output_types']}

**Required Output Structure:**

1. **Concept Summary**: 
   <Clear 100-200 word explanation of the concept>

2. **Quiz Questions**: 
   <3-5 multiple choice questions with answers>
   Format: Q1: [question]
   a) [option 1]
   b) [option 2]
   c) [option 3]
   d) [option 4]
   (Correct answer: [letter]) 

3. **Real-life Examples**: 
   <2-3 practical examples of how this concept applies in real world>

4. **Key Analogies**: 
   <1-2 simple analogies to help understand the concept>

5. **Study Tips**: 
   <3-5 bullet points with memorization techniques or study strategies>

**Rules:**
- Tailor content to {data['level']} level
- Include all requested output types
- Use simple, clear language
- Ensure all sections are completed
- Use clear section headers exactly as shown above"""

    try:
        logger.debug("Generating education content with prompt...")
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": detailed_prompt}],
            max_tokens=2000,
            temperature=0.6,
            top_p=0.9,
            repeat_penalty=1.1
        )
        content = response['choices'][0]['message']['content']
        logger.debug(f"Raw LLM response: {content}")
        return content
    except Exception as e:
        logger.error(f"Education generation error: {str(e)}")
        return f"Error: {str(e)}"

# ===== Social Media Content Generation =====
def generate_social_media_content(data):
    """Generate social media content with detailed prompt"""
    # Extract text from image if provided
    image_description = ""
    if data.get('image_base64'):
        try:
            image_data = base64.b64decode(data['image_base64'].split(',')[1])
            image = Image.open(BytesIO(image_data))
            
            # Simple image description (in a real app, you might use an image captioning model)
            image_description = "An image was provided but automatic description is not implemented. "
            
            # Optional: Use OCR to extract text from image
            try:
                extracted_text = pytesseract.image_to_string(image)
                if extracted_text.strip():
                    image_description += f"Extracted text from image: '{extracted_text[:200]}'"
            except:
                pass
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            image_description = "Could not process the uploaded image."

    detailed_prompt = f"""**Social Media Content Generation Task**

**Output Format Requirements:**
Your response MUST use EXACTLY these section headers (including colons):
1. Caption: 
2. Post Ideas: 
3. Hashtags: 

**Social Media Details:**
- Platform: {data['platform']}
- Tone: {data['tone']}
- Post Context: {data['post_context']}
- Keywords: {data.get('keywords', '')}
- Target Audience: {data.get('target_audience', '')}
- Image Description: {image_description}

**Required Output Structure:**

1. **Caption**: 
   <Engaging 1-2 sentence caption optimized for {data['platform']} with {data['tone']} tone>

2. **Post Ideas**: 
   <1-2 creative post format ideas (e.g., reel concept, carousel theme)>
   - Idea 1: [description]
   - Idea 2: [description]

3. **Hashtags**: 
   <5-10 relevant hashtags separated by spaces>
   #example #hashtags

**Rules:**
- Optimize content specifically for {data['platform']}
- Maintain {data['tone']} tone throughout
- Include relevant keywords: {data.get('keywords', 'None provided')}
- Target audience: {data.get('target_audience', 'General audience')}
- Keep captions concise (under 2200 chars for Instagram, 280 for Twitter, etc.)
- Make post ideas actionable and creative
- Use trending but relevant hashtags"""

    try:
        logger.debug("Generating social media content with prompt...")
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": detailed_prompt}],
            max_tokens=1200,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )
        content = response['choices'][0]['message']['content']
        logger.debug(f"Raw LLM response: {content}")
        return content
    except Exception as e:
        logger.error(f"Social media generation error: {str(e)}")
        return f"Error: {str(e)}"

# ===== Common Response Parser =====
def parse_response(response, section_headers):
    """Improved parsing with robust header matching and content extraction"""
    # Initialize result dictionary
    result = {header: "" for header in section_headers}
    current_section = None
    current_content = []

    # Split response into lines and process
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with a section header
        is_header = False
        for header in section_headers:
            # Match header with or without colon, case-insensitive
            if re.match(rf"^\d*\.?\s*\**\s*{re.escape(header)}\s*:?\s*", line, re.IGNORECASE):
                is_header = True
                # If we were collecting content for a previous section, save it
                if current_section and current_content:
                    result[current_section] = "\n".join(current_content).strip()
                    current_content = []
                current_section = header
                # Extract content after the header (if any)
                content_start = re.sub(rf"^\d*\.?\s*\**\s*{re.escape(header)}\s*:?\s*", "", line, flags=re.IGNORECASE)
                if content_start.strip():
                    current_content.append(content_start.strip())
                break
        if not is_header and current_section:
            # Append line to current section's content
            current_content.append(line)

    # Save the last section's content
    if current_section and current_content:
        result[current_section] = "\n".join(current_content).strip()

    # Validate and clean up
    for header in section_headers:
        if not result[header]:
            result[header] = f"Section '{header}' not found in response"
        else:
            # Remove any leading colons or numbers
            result[header] = re.sub(r"^\s*:?\s*", "", result[header]).strip()

    logger.debug(f"Parsed response: {result}")
    return result

# ===== Routes =====
@app.route('/')
def index():
    return render_template('index.html')  # Main website

@app.route('/marketing')
def marketing_form():
    return render_template('index2.html')  # Marketing form page

@app.route('/education')
def education_form():
    return render_template('education_form.html')  # Education form page

@app.route('/social-media')
def social_media_form():
    return render_template('social_media_form.html')  # Social media form page

@app.route('/generate-marketing', methods=['POST'])
def generate_marketing():
    if not request.is_json:
        logger.warning("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    required_fields = ['company_desc', 'contact_details', 'email', 'phone', 'founder_name', 'tone']
    
    if not all(field in data for field in required_fields):
        missing = [field for field in required_fields if field not in data]
        logger.warning(f"Missing required fields: {missing}")
        return jsonify({"error": f"Missing required fields: {missing}"}), 400
    
    if not llm:
        load_model()
    
    try:
        raw_response = generate_marketing_content(data)
        section_headers = ["Company Name", "Description", "Industry Type", "Tagline", "Marketing Copy"]
        structured_response = parse_response(raw_response, section_headers)
        
        logger.debug(f"Final structured response: {structured_response}")
        
        if all(v.startswith("Section") for v in structured_response.values()):
            logger.warning("Parsing failed for all sections")
            structured_response['raw_response'] = raw_response
        
        return jsonify(structured_response)
    except Exception as e:
        logger.error(f"Marketing content generation error: {str(e)}")
        return jsonify({
            "error": str(e),
            "raw_response": raw_response if 'raw_response' in locals() else None
        }), 500

@app.route('/generate-education', methods=['POST'])
def generate_education():
    if not request.is_json:
        logger.warning("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    required_fields = ['subject', 'topic', 'level', 'output_types']
    
    if not all(field in data for field in required_fields):
        missing = [field for field in required_fields if field not in data]
        logger.warning(f"Missing required fields: {missing}")
        return jsonify({"error": f"Missing required fields: {missing}"}), 400
    
    if not llm:
        load_model()
    
    try:
        raw_response = generate_education_content(data)
        section_headers = ["Concept Summary", "Quiz Questions", "Real-life Examples", "Key Analogies", "Study Tips"]
        structured_response = parse_response(raw_response, section_headers)
        
        logger.debug(f"Final structured response: {structured_response}")
        
        if all(v.startswith("Section") for v in structured_response.values()):
            logger.warning("Parsing failed for all sections")
            structured_response['raw_response'] = raw_response
        
        return jsonify(structured_response)
    except Exception as e:
        logger.error(f"Education content generation error: {str(e)}")
        return jsonify({
            "error": str(e),
            "raw_response": raw_response if 'raw_response' in locals() else None
        }), 500

@app.route('/generate-social-media', methods=['POST'])
def generate_social_media():
    if not request.is_json:
        logger.warning("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    required_fields = ['platform', 'tone', 'post_context']
    
    if not all(field in data for field in required_fields):
        missing = [field for field in required_fields if field not in data]
        logger.warning(f"Missing required fields: {missing}")
        return jsonify({"error": f"Missing required fields: {missing}"}), 400
    
    if not llm:
        load_model()
    
    try:
        raw_response = generate_social_media_content(data)
        section_headers = ["Caption", "Post Ideas", "Hashtags"]
        structured_response = parse_response(raw_response, section_headers)
        
        logger.debug(f"Final structured response: {structured_response}")
        
        if all(v.startswith("Section") for v in structured_response.values()):
            logger.warning("Parsing failed for all sections")
            structured_response['raw_response'] = raw_response
        
        return jsonify(structured_response)
    except Exception as e:
        logger.error(f"Social media content generation error: {str(e)}")
        return jsonify({
            "error": str(e),
            "raw_response": raw_response if 'raw_response' in locals() else None
        }), 500

if __name__ == '__main__':
    try:
        load_model()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise