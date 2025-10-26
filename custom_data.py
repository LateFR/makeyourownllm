import json
import argparse
import os
from pathlib import Path
from bs4 import BeautifulSoup
import re

def convert_instagram_to_training_format(input_dir, output_path, your_name):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Error: Directory '{input_dir}' not found")
        return

    total_messages = 0
    conversations_count = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        for person_dir in sorted(input_path.iterdir()):
            if not person_dir.is_dir():
                continue
            print(f"📂 Processing conversation: {person_dir.name}")
            conversations_count += 1

            for message_file in sorted(person_dir.glob("message_*.json")):
                try:
                    with open(message_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    messages = data.get('messages', [])
                    messages.reverse()
                    for message in messages:
                        sender = message.get('sender_name', 'Unknown')
                        content = message.get('content', '').strip()
                        if content:
                            prefix = "user: " if sender != your_name else "assistant: "
                            out.write(prefix + content + '\n')
                            total_messages += 1
                except Exception as e:
                    print(f"⚠️ Warning: Could not process {message_file}: {e}")

    print(f"✅ Success! {total_messages} messages from {conversations_count} conversations written to {output_path}")


def convert_html_to_training_format(input_dir, output_path, chunk_size_mb=5):
    """
    Lazy-loads large HTML files (>5MB) to avoid memory overflow.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Error: Directory '{input_dir}' not found")
        return

    total_files = 0
    total_text_length = 0
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB → bytes

    with open(output_path, 'w', encoding='utf-8') as out:
        html_files = list(input_path.rglob("*.html"))
        if not html_files:
            print(f"⚠️ Warning: No HTML files found in {input_dir}")
            return

        for html_file in sorted(html_files):
            size_mb = os.path.getsize(html_file) / (1024 * 1024)
            print(f"📄 Processing: {html_file.name} ({size_mb:.1f} MB)")

            try:
                if size_mb > chunk_size_mb:
                    print(f"⚙️ Large file detected → using lazy mode ({chunk_size_mb} MB chunks)")
                    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                        buffer = ""
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            buffer += chunk

                            # Parse only up to last complete tag
                            last_tag = buffer.rfind(">")
                            if last_tag != -1:
                                partial_html = buffer[:last_tag+1]
                                buffer = buffer[last_tag+1:]
                                soup = BeautifulSoup(partial_html, 'html.parser')
                                text = soup.get_text(separator='\n', strip=True)
                                for line in text.split('\n'):
                                    if line.strip():
                                        out.write(line.strip() + '\n')
                                        total_text_length += len(line)
                else:
                    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                        soup = BeautifulSoup(f, 'html.parser')
                        text = soup.get_text(separator='\n', strip=True)
                        for line in text.split('\n'):
                            if line.strip():
                                out.write(line.strip() + '\n')
                                total_text_length += len(line)

                total_files += 1

            except Exception as e:
                print(f"⚠️ Warning: Could not process {html_file}: {e}")

    print(f"✅ Success! {total_files} HTML files processed, {total_text_length} characters written to {output_path}")
def stream_find_json_array(fp, start_marker="var jsonData"):
    """
    Recherche le début de la variable JS 'var jsonData = [' puis renvoie
    la position où commence le tableau JSON.
    """
    CHUNK = 1024 * 1024  # 1 MB
    buffer = ""
    
    while True:
        chunk = fp.read(CHUNK)
        if not chunk:
            return None
        
        buffer += chunk
        
        # Chercher "var jsonData = ["
        m = re.search(rf"{re.escape(start_marker)}\s*=\s*\[", buffer)
        if m:
            # Trouver la position exacte du '['
            bracket_pos = buffer.find('[', m.start())
            if bracket_pos != -1:
                # Garder ce qui reste après le '['
                remaining = buffer[bracket_pos:]
                return remaining, fp
        
        # Garder les derniers caractères pour éviter de couper le pattern
        if len(buffer) > 10 * CHUNK:
            buffer = buffer[-2 * CHUNK:]
    
    return None


def extract_json_array_from_large_file(path, start_marker="var jsonData", chunk_size=1024*1024):
    """
    Extrait le tableau JSON complet en streaming, même s'il fait des centaines de MB.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        found = stream_find_json_array(f, start_marker=start_marker)
        if not found:
            return None
        
        remaining, fp = found
        
        # État du parser
        result = []
        bracket_level = 0
        in_string = False
        escape = False
        started = False
        
        # Buffer circulaire pour traiter character par character
        def char_generator():
            """Génère les caractères un par un depuis remaining + file"""
            for c in remaining:
                yield c
            while True:
                chunk = fp.read(chunk_size)
                if not chunk:
                    break
                for c in chunk:
                    yield c
        
        char_stream = char_generator()
        
        for c in char_stream:
            # Attendre le premier '['
            if not started:
                if c == '[':
                    started = True
                    bracket_level = 1
                    result.append(c)
                continue
            
            # Parser JSON caractère par caractère
            if escape:
                escape = False
                result.append(c)
                continue
            
            if c == '\\':
                escape = True
                result.append(c)
            elif c == '"':
                in_string = not in_string
                result.append(c)
            elif not in_string:
                if c == '[' or c == '{':
                    bracket_level += 1
                    result.append(c)
                elif c == ']' or c == '}':
                    bracket_level -= 1
                    result.append(c)
                    
                    # Fin du tableau principal détectée
                    if bracket_level == 0 and c == ']':
                        return "".join(result)
                else:
                    result.append(c)
            else:
                result.append(c)
        
        # Si on arrive ici, le fichier s'est terminé sans fermer le tableau
        return "".join(result) if started else None


def parse_chatgpt_json_and_write(json_obj, out_fp):
    """
    Parse la structure ChatGPT et écrit au format user:/assistant:
    """
    written = 0
    
    # Le JSON est une liste de conversations
    if not isinstance(json_obj, list):
        json_obj = [json_obj]
    
    for conv in json_obj:
        if not isinstance(conv, dict):
            continue
        
        # Extraire les messages depuis 'mapping'
        mapping = conv.get("mapping", {})
        if not mapping:
            continue
        
        # Construire une liste ordonnée de messages
        messages = []
        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            
            msg = node.get("message")
            if not msg or not isinstance(msg, dict):
                continue
            
            # Extraire les infos importantes
            create_time = msg.get("create_time", 0)
            author_role = msg.get("author", {}).get("role", "unknown")
            content = msg.get("content", {})
            
            # Extraire le texte
            if isinstance(content, dict):
                parts = content.get("parts", [])
                if isinstance(parts, list) and parts:
                    text = "\n".join([str(p) for p in parts if p]).strip()
                    if text:
                        messages.append({
                            "time": create_time,
                            "role": author_role,
                            "text": text
                        })
        
        # Trier par temps
        messages.sort(key=lambda m: m["time"])
        
        # Écrire les messages
        for msg in messages:
            role = msg["role"].lower()
            
            # Ignorer les messages système
            if role == "system" or role == "tool":
                continue
            
            # Déterminer le préfixe
            if role == "user":
                prefix = "user: "
            elif role == "assistant":
                prefix = "assistant: "
            else:
                prefix = "user: "  # fallback
            
            out_fp.write(prefix + msg["text"] + "\n")
            written += 1
    
    return written


def convert_chatgpt_html_to_training_format(input_dir, output_path, chunk_size_mb=5):
    """
    Mode 'chatgpt': extrait jsonData de chaque HTML export ChatGPT.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Error: Directory '{input_dir}' not found")
        return

    total_files = 0
    total_written = 0
    chunk_bytes = int(chunk_size_mb * 1024 * 1024)

    html_files = list(input_path.rglob("*.html"))
    if not html_files:
        print(f"⚠️  Warning: No HTML files found in {input_dir}")
        return

    with open(output_path, "w", encoding="utf-8") as out:
        for html_file in sorted(html_files):
            try:
                size_mb = os.path.getsize(html_file) / (1024 * 1024)
                print(f"📄 Processing: {html_file.name} ({size_mb:.1f} MB)")

                # Toujours utiliser le mode streaming pour les gros fichiers
                json_text = extract_json_array_from_large_file(
                    html_file, 
                    start_marker="var jsonData",
                    chunk_size=chunk_bytes
                )

                if not json_text or not json_text.strip():
                    print(f"⚠️  Warning: Could not extract jsonData from {html_file.name}")
                    total_files += 1
                    continue

                # Parser le JSON
                print(f"🔍 Parsing JSON ({len(json_text) / 1024 / 1024:.1f} MB)...")
                try:
                    parsed = json.loads(json_text)
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON parsing failed: {e}")
                    # Tentative de réparation simple
                    json_text = re.sub(r",\s*]", "]", json_text)
                    json_text = re.sub(r",\s*}", "}", json_text)
                    try:
                        parsed = json.loads(json_text)
                    except:
                        print(f"❌ Could not parse JSON from {html_file.name}")
                        total_files += 1
                        continue

                # Écrire les conversations
                written = parse_chatgpt_json_and_write(parsed, out)
                print(f"✅ {html_file.name}: {written} messages extracted")
                total_written += written
                total_files += 1

            except Exception as e:
                print(f"⚠️  Error processing {html_file.name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n✅ Done! {total_files} files processed, {total_written} messages written to {output_path}")

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
                       choices=['instagram', 'html', 'chatgpt'],
                       help="Type of data to convert")
    parser.add_argument("--your-name", type=str, default=None,
                       help="Your name as it appears in Instagram (required for instagram type)")
    parser.add_argument("--chunk-size-mb", type=float, default=5.0,
                       help="Chunk size in MB for ChatGPT conversion (default: 5.0)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    if args.type == 'instagram':
        if not args.your_name:
            print("❌ Error: --your-name is required for Instagram conversion")
            exit(1)
        print(f"🔄 Converting Instagram messages from {args.input_path}...")
        convert_instagram_to_training_format(args.input_path, args.output_path, args.your_name)

    elif args.type == 'html':
        print(f"🔄 Extracting text from HTML files in {args.input_path}...")
        convert_html_to_training_format(args.input_path, args.output_path)
    elif args.type == 'chatgpt':
        print(f"🔄 Extracting ChatGPT jsonData from HTML files in {args.input_path}...")
        convert_chatgpt_html_to_training_format(args.input_path, args.output_path, chunk_size_mb=args.chunk_size_mb)
