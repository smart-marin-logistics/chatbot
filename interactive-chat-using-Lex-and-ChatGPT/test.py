import os
import glob
directory_path = './interactive-chat-using-Lex-and-ChatGPT/pdf_files'
if os.path.exists(directory_path):
    print(f"Directory '{directory_path}' exists.")
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(pdf)
    else:
        print(f"No PDF files found in '{directory_path}'.")
else:
    print(f"Directory '{directory_path}' does not exist.")
