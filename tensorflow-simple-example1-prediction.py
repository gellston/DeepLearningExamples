import tensorflow as tf
import datasetloader as loader


loader = loader.datasetloader('/animal', loader.pathtype.relative)
classCount = loader.label_count()

sess = tf.Session()
saver = tf.train.import_meta_graph('./pre-trained-model/animal.model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./pre-trained-model'))


graph = tf.get_default_graph()
Output = graph.get_tensor_by_name("output:0")
Input = graph.get_tensor_by_name("Input:0")

result = [0, 0, 0]

image, labels = loader.load([30000], 255, 3)
feed_dict = {Input: image}

prediction = sess.run(Output, feed_dict)

print ('=== label values ===')
print (labels)
print ('=== prediction values ===')
print(prediction)
