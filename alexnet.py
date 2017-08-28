from network import Network
import tensorflow as tf

class CaffeNetPlaces365(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 12, 1, 1, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(3, 3, 24, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 48, 1, 1, name='conv3')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(36, name='fc6')
             .fc(36, relu=False, name='fc7')
             .softmax(name='prob'))

def test():
    #input_node = tf.placeholder(tf.float32, shape=(None, 227,227,3))
    input_node = tf.random_uniform(shape=(1,227,227,3))*255
    net = CaffeNetPlaces365({'data': input_node})

    c = tf.ConfigProto()
    c.gpu_options.visible_device_list="0"

    with tf.Session(config=c) as sess:
        print('Loading the model')
        net.load('../../models/alexnet.npy', sess)
        temp = sess.run(net.layers['prob'])
        print (temp)
        print(temp.shape)
        print('Done!')

if __name__ == '__main__':
    test()
