# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def extract_previous_messages(chat_messages_array):
    messages = "\n"
    for msg in chat_messages_array[:-1]:
        messages += (msg["role"] + ": " + msg["content"] + "\n")
    return messages

def combine_all_messages_for_vector_search(chat_messages_array):
    return extract_previous_messages(chat_messages_array) + extract_user_query_string(chat_messages_array)

# Method to format the docs returned by the retriever into the prompt
def format_context(docs):
    chunk_template = '''{chunk_text}
    
    '''
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
        )
        for d in docs
    ]
    return "".join(chunk_contents)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)