import tensorflow as tf 
import numpy as np 

class RBM():
    def __init__(self, nv=28*28, nh=512, cd_steps=3):
        self.graph = tf.Graph() 
        with self.graph.as_default(): 
            self.W = tf.Variable(tf.random.truncated_normal((nv, nh)) * 0.01)
            self.bv = tf.Variable(tf.zeros((nv, 1))) 
            self.bh = tf.Variable(tf.zeros((nh, 1)))
            
            self.cd_steps = cd_steps 
            self.modelW = None 
    
    def bernoulli(self, p):
        return tf.nn.relu(tf.sign(p - tf.compat.v1.random_uniform(p.shape)))
    
    def energy(self, v):
        b_term = tf.matmul(v, self.bv)
        linear_transform = tf.matmul(v, self.W) + tf.squeeze(self.bh)
        h_term = tf.reduce_sum(tf.compat.v1.log(tf.exp(linear_transform) + 1), axis=1) 
        return tf.reduce_mean(-h_term -b_term)
    
    def sample_h(self, v):
        ph_given_v = tf.sigmoid(tf.matmul(v, self.W) + tf.squeeze(self.bh))
        return self.bernoulli(ph_given_v)
    
    def sample_v(self, h):
        pv_given_h = tf.sigmoid(tf.matmul(h, tf.transpose(self.W)) + tf.squeeze(self.bv))
        return self.bernoulli(pv_given_h)
    
    def gibbs_step(self, i, k, vk):
        hk = self.sample_h(vk)
        vk = self.sample_v(hk)
        return i+1, k, vk
    
    def train(self, X, lr=0.001, batch_size=64, epochs=5):
        with self.graph.as_default(): 
            tf_v = tf.compat.v1.placeholder(tf.float32, [batch_size, self.bv.shape[0]])
            v = tf.round(tf_v) 
            vk = tf.identity(v)

            i = tf.constant(0)
            k = tf.constant(self.cd_steps)
            _, _, vk = tf.while_loop(cond = lambda i, k, *args: i <= k,
                                      body = self.gibbs_step,
                                      loop_vars = [i, k, vk],
                                      parallel_iterations=1,
                                      back_prop=False)

            vk = tf.stop_gradient(vk) 
            energy_v = self.energy(v)
            energy_vk = self.energy(vk)
            loss = energy_v - energy_vk
            optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)
            init = tf.compat.v1.global_variables_initializer()
        
        all_losses = []
        with tf.compat.v1.Session(graph=self.graph) as sess:
            init.run()
            for epoch in range(epochs): 
                losses = []
                for i in range(0, len(X)-batch_size, batch_size):
                    x_batch = X[i:i+batch_size] 
                    _,l = sess.run([optimizer,loss], feed_dict={tf_v: x_batch})
                    losses.append(l)

                self.model_bh = np.squeeze(self.bh.eval())
                self.model_bv = np.squeeze(self.bv.eval())
                self.model_W = self.W.eval()

                mean_loss = np.mean(losses)
                all_losses.append(mean_loss)
                print('Epoch Cost %d: ' % epoch, mean_loss, end='\r')
            return all_losses


