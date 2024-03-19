import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import datetime
import numpy as np

from CBOW import *
from Logging import *


NUM_EPOCHS = 20
BATCH_SIZE = 64

window_size = 3
embedding_size = 256
num_sampled_negative_classes = 32


def load_data():
    with open("./data/Harry_Potter_all_books_preprocessed.txt") as file:
        content = file.read()

    content = content.lower()
    content = content.replace("!", " ")
    content = content.replace("?", " ")
    content = content.strip()
    sentences = content.split(".")

    sentences_words = [sentence.split(" ")[:-1] for sentence in sentences]

    sentences_words = [sentence for sentence in sentences_words if 2*window_size + 1 < len(sentence)]

    return sentences_words


def dataset_generator():

    sentences_words = load_data()
    
    dict_word_token = {}
    dict_word_cnt = {}
    num_total_words = 0

    cnt_tokens = 0
    for sentence in sentences_words:
        for word in sentence:
            
            if word not in dict_word_token:
                dict_word_token[word] = cnt_tokens
                cnt_tokens += 1

            if word not in dict_word_cnt:
                dict_word_cnt[word] = 1
            else:
                dict_word_cnt[word] += 1

            num_total_words += 1

    for sentence in sentences_words:
  
        if 2*window_size + 1 > len(sentence):
            continue
        
        for target_word_idx in range(window_size, len(sentence) - window_size):
            
            target_word = sentence[target_word_idx]

            #
            # Subsampling
            #
            s = 0.001
            z_w = dict_word_cnt[target_word] / num_total_words
            p_keep = ( np.sqrt(z_w/s) + 1 ) * (s/z_w)
            rand_var = np.random.uniform() # [0, 1)
            
            if p_keep > rand_var:

                context_words = sentence[target_word_idx-window_size:target_word_idx + window_size + 1]
                context_words.pop(window_size)

                context_words_token = [dict_word_token[context_word] for context_word in context_words]
                
                
                target_word_token = dict_word_token[target_word]

                pair = (np.array(context_words_token), np.array([target_word_token]))
           
                yield pair


def main():

    dict_word_token = {}
  
    sentences_words = load_data()

    cnt_tokens = 0
    for sentence in sentences_words:
        for word in sentence:
            
            if word not in dict_word_token:
                dict_word_token[word] = cnt_tokens
                cnt_tokens += 1

    
    #dict_token_word = {token: word for work, token in dict_word_token.items()}

    vocab_size = len(dict_word_token)

    dataset = tf.data.Dataset.from_generator(
                    dataset_generator,
                    output_signature=(
                            tf.TensorSpec(shape=(2*window_size,), dtype=tf.uint16),
                            tf.TensorSpec(shape=(1,), dtype=tf.uint16)
                        )
                )
    
    #dataset = dataset.take(200000)
    dataset = dataset.apply(prepare_data)

   
    train_dataset = dataset.skip(750)
    test_dataset = dataset.take(750)

    #
    # Logging
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)
    


    #
    # Initialize model
    #
    cbow = CBOW(vocab_size, embedding_size)
    cbow.build(input_shape=(None, vocab_size))
    cbow.summary()
    
    #
    # Train and test loss/accuracy
    #
    print(f"Epoch 0")
  
    log(train_summary_writer, cbow, train_dataset, test_dataset, dict_word_token, num_sampled_negative_classes, epoch=0)
 
    #
    # Train loop
    #
    for epoch in range(1, NUM_EPOCHS + 1):
            
        print(f"Epoch {epoch}")

        for context_words_token, target_word_token in tqdm.tqdm(train_dataset, position=0, leave=True): 
            cbow.train_step(context_words_token, target_word_token, num_sampled_negative_classes)

        log(train_summary_writer, cbow, train_dataset, test_dataset, dict_word_token, num_sampled_negative_classes, epoch)

        # Save model (its parameters)
        if epoch % 5 == 0:
            cbow.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")



def prepare_data(dataset):

    # Cache
    dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(50000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")