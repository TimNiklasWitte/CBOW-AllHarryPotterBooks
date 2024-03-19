import tensorflow as tf

class CBOW(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size):
        super(CBOW, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
    
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
   

        self.metric_loss = tf.keras.metrics.Mean(name="loss")

    def build(self, input_shape):
        
        input_shape_vocab_size = input_shape[1]
        assert(input_shape_vocab_size == self.vocab_size)

        self.w_score = self.add_weight(
            shape=(self.vocab_size, self.embedding_size), 
            initializer="random_normal", 
            trainable=True,
            name='w_score'
        )

        self.b_score = self.add_weight(
            shape=(self.vocab_size, ), 
            initializer="random_normal", 
            trainable=True,
            name='b_score'
        )


        super(CBOW, self).build(input_shape)
        

    @tf.function
    def call(self, context_tokens):
        mean_embedding = self.get_mean_embedding(context_tokens)

        y = mean_embedding @ tf.transpose(self.w_score, perm=(1,0))  + self.b_score

        return y
    
    @tf.function
    def get_mean_embedding(self, context_tokens):
        embeddings = self.embedding_layer(context_tokens)
        mean_embedding = tf.math.reduce_mean(embeddings, axis=1)

        return mean_embedding

    @tf.function
    def train_step(self, context_tokens, target_token, num_sampled_negative_classes):

        with tf.GradientTape() as tape:
            mean_embedding = self.get_mean_embedding(context_tokens)
            
            loss = tf.nn.nce_loss(
                        weights=self.w_score,                      # [vocab_size, embed_size]
                        biases=self.b_score,                       # [vocab_size]
                        labels=target_token,                       # [batch_size, 1]
                        inputs=mean_embedding,                     # [batch_size, embed_size]
                        num_sampled=num_sampled_negative_classes,  # negative sampling: number 
                        num_classes=self.vocab_size,
                        num_true=1                                 # positive sample
                    )

            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)



    def test_step(self, dataset, num_sampled_negative_classes):
          
        self.metric_loss.reset_states()
 
        for context_tokens, target_token in dataset:
            mean_embedding = self.get_mean_embedding(context_tokens)
            
            loss = tf.nn.nce_loss(
                        weights=self.w_score,                      # [vocab_size, embed_size]
                        biases=self.b_score,                       # [vocab_size]
                        labels=target_token,                       # [batch_size, 1]
                        inputs=mean_embedding,                     # [batch_size, embed_size]
                        num_sampled=num_sampled_negative_classes,  # negative sampling: number 
                        num_classes=self.vocab_size,
                        num_true=1                                 # positive sample
                    )
            
            loss = tf.reduce_mean(loss)

            self.metric_loss.update_state(loss)

