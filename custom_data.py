import json
import argparse
import os
from pathlib import Path
from bs4 import BeautifulSoup

def convert_instagram_to_training_format(input_dir, output_path, your_name):
    """
    Convert Instagram message export to training format.
    Structure: input_dir/person_name/message_1.json, message_2.json, etc.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Error: Directory '{input_dir}' not found")
        return
    
    total_messages = 0
    conversations_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as out:
        # Parcourir tous les sous-dossiers (une personne = un dossier)
        for person_dir in sorted(input_path.iterdir()):
            if not person_dir.is_dir():
                continue
            
            print(f"📂 Processing conversation: {person_dir.name}")
            conversations_count += 1
            
            # Trouver tous les message_N.json dans l'ordre
            message_files = sorted(person_dir.glob("message_*.json"))
            
            for message_file in message_files:
                try:
                    with open(message_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Instagram structure: {"messages": [...], "participants": [...]}
                    messages = data.get('messages', [])
                    
                    # Les messages sont en ordre inverse (plus récent en premier)
                    messages.reverse()
                    
                    for message in messages:
                        sender = message.get('sender_name', 'Unknown')
                        content = message.get('content', '').strip()
                        
                        if content:
                            # Déterminer si c'est toi ou l'assistant
                            if sender != your_name:
                                prefix = "user: "
                            else:
                                prefix = "assistant: "
                            
                            out.write(prefix + content + '\n')
                            total_messages += 1
                
                except Exception as e:
                    print(f"⚠️  Warning: Could not process {message_file}: {e}")
    
    print(f"✅ Success! {total_messages} messages from {conversations_count} conversations written to {output_path}")

def convert_html_to_training_format(input_dir, output_path):
    """
    Extract raw text from all HTML files in a directory.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Error: Directory '{input_dir}' not found")
        return
    
    total_files = 0
    total_text_length = 0
    
    with open(output_path, 'w', encoding='utf-8') as out:
        # Trouver tous les fichiers .html
        html_files = list(input_path.rglob("*.html"))
        
        if not html_files:
            print(f"⚠️  Warning: No HTML files found in {input_dir}")
            return
        
        for html_file in sorted(html_files):
            print(f"📄 Processing: {html_file.name}")
            
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                
                # Extraire tout le texte brut
                text = soup.get_text(separator='\n', strip=True)
                
                # Nettoyer les lignes vides multiples
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for line in lines:
                    out.write(line + '\n')
                    total_text_length += len(line)
                
                total_files += 1
            
            except Exception as e:
                print(f"⚠️  Warning: Could not process {html_file}: {e}")
    
    print(f"✅ Success! {total_files} HTML files processed, {total_text_length} characters written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert custom data to training format",
        epilog="""Examples:
  Instagram: python custom_data.py instagram_messages/ data/insta.txt -t instagram --your-name "Ton Nom"
  HTML:      python custom_data.py html_folder/ data/html.txt -t html
        """
    )
    parser.add_argument("input_path", type=str, help="Path to input directory")
    parser.add_argument("output_path", type=str, help="Path to output file")
    parser.add_argument("-t", "--type", type=str, required=True,
                       choices=['instagram', 'html'],
                       help="Type of data to convert")
    parser.add_argument("--your-name", type=str, default=None,
                       help="Your name as it appears in Instagram (required for instagram type)")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    
    # Convert based on type
    if args.type == 'instagram':
        if not args.your_name:
            print("❌ Error: --your-name is required for Instagram conversion")
            print("Example: python custom_data.py messages/ output.txt -t instagram --your-name 'John Doe'")
            exit(1)
        
        print(f"🔄 Converting Instagram messages from {args.input_path}...")
        print(f"👤 Your name: {args.your_name}")
        convert_instagram_to_training_format(args.input_path, args.output_path, args.your_name)
    
    elif args.type == 'html':
        print(f"🔄 Extracting text from HTML files in {args.input_path}...")
        convert_html_to_training_format(args.input_path, args.output_path)