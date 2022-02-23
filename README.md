# pointer network to rank ten random numbers which will be used in stock market


Environment:
MMY algorithms, there are three algorithms, including OME,BMP, BOMP

Usage:
 use data_produce to produce the training data and test data
 use rnn_pointer to train and test the model of rank random numbers which will be used to rank ten kinds of stocks in stock market.
 
Network:

class Encoder(nn.Module):

    
    def __init__(self, input_size, hidden_size, batch_size, bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False, bidirectional=bidirectional)

    def forward(self, inputs, hidden):
        output, hidden = self.lstm(inputs, hidden)   
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), self.batch_size, self.hidden_size),
                torch.zeros(1 + int(self.bidirectional), self.batch_size, self.hidden_size))   #(num_layers * num_directions, batch, hidden_size)


class AttentionDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, batch_size, vocab_size,seq_len):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.attn = nn.Linear(hidden_size + output_size + vocab_size, 1)    
        self.lstm = nn.LSTM(hidden_size + vocab_size, output_size)    
        self.final = nn.Linear(output_size, vocab_size)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.output_size),
                torch.zeros(1, self.batch_size, self.output_size))

    
    def forward(self, decoder_hidden, encoder_outputs, input):
        seq = 0
        weights= []
        i = 0
        output = torch.zeros(self.batch_size, self.vocab_size)
        for i in range(len(encoder_outputs)):
            weights.append(self.attn(torch.cat((decoder_hidden[0][:].squeeze(0),encoder_outputs[i],output), dim=1)))

        normalized_weight = F.softmax(torch.cat(weights, 1), 1)
        normalized_weights = normalized_weight


        attn_applied = torch.bmm(normalized_weight.unsqueeze(1),
                                 encoder_outputs.transpose(0,1))  
        input_lstm = torch.cat((attn_applied.transpose(0,1)[0], output),
                               dim=1)  # if we are using embedding, use embedding of input here instead

        output_, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)

        output = self.final(output_[0]) #output 为（vocab_size, output_size）
        #output = self.final2(output)

        # hidden0 = hidden[0].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
        # hidden1 = hidden[1].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
        # decoder_hidden = (hidden0, hidden1)
        # decoder_hiddens = decoder_hidden


        out = F.softmax(output,1)

        return out
