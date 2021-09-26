import numpy as np
import scratch_grad as sg


class ScratchNet:
    def __init__(self, inp_size=10, hidden_size=20, output_size=3):
        self.sg_w1 = sg.Variable((inp_size, hidden_size))
        self.sg_w1.value = np.random.normal(size=(inp_size,hidden_size))

        self.sg_w2 = sg.Variable((hidden_size, hidden_size))
        self.sg_w2.value = np.random.normal(size=(hidden_size, hidden_size))

        self.sg_w3 = sg.Variable((hidden_size, output_size))
        self.sg_w3.value = np.random.normal(size=(hidden_size, output_size))

        self.params = [self.sg_w1, self.sg_w2, self.sg_w3]

    def forward(self, x):
        sg_layer1_out = sg.relu(x @ self.sg_w1)
        sg_layer2_out = sg.relu(sg_layer1_out @ self.sg_w2)
        sg_layer3_out = sg_layer2_out @ self.sg_w3

        return sg_layer3_out

    def zero_grad(self):
        for p in self.params:
            p.gradient = None

            # clean-up the references from the previous forward-pass
            p.consumer_operations = []


if __name__ == '__main__':

    train_set_size = 16
    inp_size = 10
    hidden_size = 10
    output_size = 5
    epochs = 20
    learning_rate = 0.01

    net = ScratchNet(inp_size, hidden_size, output_size)

    # Random input
    train_data = np.random.normal(size=(train_set_size, inp_size))
    
    # Random ground truth labels
    ground_truth = np.random.randint(0, output_size, train_set_size) 

    np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.5f}'.format})    

    for i in range(epochs):
        print("Epoch:", i + 1)

        losses = []


        total_loss = sg.Variable(())
        total_loss.value = np.zeros(shape=())

        n_correct = 0

        for j in range(train_set_size):
            sg_net_in = sg.Variable((7,))
            sg_net_in.value = train_data[j]
            out = net.forward(sg_net_in)
            loss_one = sg.softmax_loss(out, ground_truth[j])

            probs = loss_one.operation.softmax_output
            print("ScratchNet output: ", loss_one.operation.softmax_output)
            print("ground_truth: ", ground_truth[j])

            if np.argmax(probs) == ground_truth[j]:
                n_correct += 1

            total_loss = total_loss + loss_one

        print(f"train loss={total_loss.value}")
        print(f"accuracy={n_correct / train_set_size * 100}%\n\n")
        total_loss.backprop()

        # apply gradients
        for p in net.params:
            p.value -= p.gradient*learning_rate

        net.zero_grad()        