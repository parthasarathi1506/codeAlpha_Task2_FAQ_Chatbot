from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = [
    {
        "question": "What is cybersecurity?",
        "answer": "Cybersecurity is the practice of protecting computers, networks, and data from digital attacks."
    },
    {
        "question": "What is phishing?",
        "answer": "Phishing is a cyber attack where attackers use fake messages or websites to steal information."
    },
    {
        "question": "What is a firewall?",
        "answer": "A firewall is a security system that monitors and controls network traffic."
    },
    {
        "question": "What is malware?",
        "answer": "Malware is malicious software designed to damage systems or steal information."
    },
    {
        "question": "How can I create a strong password?",
        "answer": "Use a long password containing letters, numbers, and symbols. Avoid reusing passwords."
    },
    {
        "question": "What is two factor authentication?",
        "answer": "Two-factor authentication adds an extra layer of security to your account."
    },
    {
        "question": "What is antivirus software?",
        "answer": "Antivirus software detects and removes malicious software from computers."
    },
    {
        "question": "How can I protect my computer?",
        "answer": "Keep software updated, use antivirus software, and avoid suspicious links."
    },
    {
        "question": "What is ransomware?",
        "answer": "Ransomware is malware that locks files and demands payment."
    },
    {
        "question": "What is a VPN?",
        "answer": "A VPN creates an encrypted connection to help protect your internet privacy."
    }
]

questions = [faq["question"] for faq in faqs]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

print("\n===================================")
print("   CYBERSECURITY FAQ CHATBOT")
print("===================================")
print("Ask any cybersecurity question!")
print("Type 'exit' to close the chatbot.\n")

while True:
    user_question = input("You: ")

    if user_question.lower() == "exit":
        print("Bot: Thank you for using the chatbot!")
        break

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        user_vector, question_vectors
    )

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score > 0.15:
        print("Bot:", faqs[best_match]["answer"])
    else:
        print("Bot: Sorry, I don't know the answer to that question.")

    print()