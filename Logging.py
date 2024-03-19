
import tensorflow as tf
import numpy as np

def log(train_summary_writer, cbow, train_dataset, test_dataset, dict_word_token, num_sampled_negative_classes, epoch):

    # Epoch 0 = no training steps are performed 
    # test based on train data
    # -> Determinate initial train_loss and train_accuracy
    if epoch == 0:
        cbow.test_step(train_dataset.take(500), num_sampled_negative_classes)

    #
    # Train
    #
    train_loss = cbow.metric_loss.result()
    cbow.metric_loss.reset_states()


    #
    # Test
    #

    cbow.test_step(test_dataset, num_sampled_negative_classes)

    test_loss = cbow.metric_loss.result()
    cbow.metric_loss.reset_states()


    #
    # Write to TensorBoard
    #
    with train_summary_writer.as_default():
        tf.summary.scalar(f"train_loss", train_loss, step=epoch)
        tf.summary.scalar(f"test_loss", test_loss, step=epoch)


    #
    # Output
    #
    print(f"train_loss: {train_loss}")
    print(f"test_loss: {test_loss}")


    words = ["hogwarts", 
             "muggle", 
             "horcrux", 
             "spell", 
             "voldemort", 
             "witch", 
             "wizard",
             "gryffindor", 
             "potion",
             "dementor",
             "patronus",
             "quidditch"]
    
    dict_word_embedding = {}
    for word, token in dict_word_token.items():
        dict_word_embedding[word] = cbow.embedding_layer(token).numpy()


    with train_summary_writer.as_default():
        for word in words:
            similar_words_cosine, sims_cosine, similar_words_distance, sims_distance = get_similar_words(cbow, word, 50, dict_word_token, dict_word_embedding)

            log_header =  "| Word (1) | cos sim (1) | Word (2) | distance (2) |\n"
            log_header += "|----------|-------------|----------|----------|\n"

            log = ""
            for idx in range(len(similar_words_cosine)):
                log += f"| {similar_words_cosine[idx]} | {sims_cosine[idx]:3.5f} | {similar_words_distance[idx]} | {sims_distance[idx]:3.5f} |\n"
                    
            log = log_header + log 
            tf.summary.text(name=word, data = log, step=epoch)
         

def cos_sim(a,b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


def get_similar_words(cbow, target_word, k, dict_word_token, dict_word_embedding):
    
    similar_words_cosine = [""] * k
    sims_cosine = [-1] * k

    similar_words_distance = [""] * k
    sims_distance = [99999999] * k

    target_word_token = dict_word_token[target_word]
    embedding_target = cbow.embedding_layer(target_word_token).numpy()
    for word, embedding_word in dict_word_embedding.items():

        if target_word != word:
  
            sim = cos_sim(embedding_target, embedding_word)
            dist = np.linalg.norm(embedding_target - embedding_word)

            if sim > np.min(sims_cosine):
                sims_cosine[np.argmin(sims_cosine)] = sim 
                      
                similar_words_cosine[np.argmin(sims_cosine)] = word 

            if dist < np.max(sims_distance):
                sims_distance[np.argmax(sims_distance)] = dist 
                similar_words_distance[np.argmax(sims_distance)] = word 

    
    # sort
    similar_words_cosine = np.array(similar_words_cosine)
    similar_words_distance = np.array(similar_words_distance)

    idxs = np.argsort(sims_cosine)[::-1] # reverse order
    sims_cosine = np.sort(sims_cosine)[::-1] # reverse order
    similar_words_cosine = similar_words_cosine[idxs]

    idxs = np.argsort(sims_distance)
    sims_distance = np.sort(sims_distance)
    similar_words_distance = similar_words_distance[idxs]

    return similar_words_cosine, sims_cosine, similar_words_distance, sims_distance