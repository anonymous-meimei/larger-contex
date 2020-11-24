"""class implements Conditional Random Fields (CRF)"""
import torch
import torch.nn as nn
from src.layers.layer_base import LayerBase
from src.classes.utils import log_sum_exp


class LayerCRF(LayerBase):
    """LayerCRF implements Conditional Random Fields (Ma.et.al., 2016 style)"""
    def __init__(self, gpu, states_num, pad_idx, sos_idx, tag_seq_indexer, verbose=True):
        super(LayerCRF, self).__init__(gpu)
        self.states_num = states_num
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.tag_seq_indexer = tag_seq_indexer
        self.tag_seq_indexer.add_tag('<sos>')
        self.verbose = verbose
        # Transition matrix contains log probabilities from state j to state i
        self.transition_matrix = nn.Parameter(torch.zeros(states_num, states_num, dtype=torch.float))
        nn.init.normal_(self.transition_matrix, -1, 0.1)
        # Default initialization
        self.transition_matrix.data[self.sos_idx, :] = -9999.0
        self.transition_matrix.data[:, self.pad_idx] = -9999.0
        self.transition_matrix.data[self.pad_idx, :] = -9999.0
        self.transition_matrix.data[self.pad_idx, self.pad_idx] = 0.0

    def get_empirical_transition_matrix(self, tag_sequences_train, tag_seq_indexer=None):
        if tag_seq_indexer is None:
            tag_seq_indexer = self.tag_seq_indexer
        empirical_transition_matrix = torch.zeros(self.states_num, self.states_num, dtype=torch.long)
        for tag_seq in tag_sequences_train:
            s = tag_seq_indexer.item2idx_dict[tag_seq[0]]
            empirical_transition_matrix[s, self.sos_idx] += 1
            for n, tag in enumerate(tag_seq):
                if n + 1 >= len(tag_seq):
                    break
                next_tag = tag_seq[n + 1]
                j = tag_seq_indexer.item2idx_dict[tag]
                i = tag_seq_indexer.item2idx_dict[next_tag]
                empirical_transition_matrix[i, j] += 1
        return empirical_transition_matrix

    def init_transition_matrix_empirical(self, tag_sequences_train):
        # Calculate statistics for tag transitions
        empirical_transition_matrix = self.get_empirical_transition_matrix(tag_sequences_train)
        # Initialize
        for i in range(self.tag_seq_indexer.get_items_count()):
            for j in range(self.tag_seq_indexer.get_items_count()):
                if empirical_transition_matrix[i, j] == 0:
                    self.transition_matrix.data[i, j] = -9999.0
                #self.transition_matrix.data[i, j] = torch.log(empirical_transition_matrix[i, j].float() + 10**-32)
        if self.verbose:
            print('Empirical transition matrix from the train dataset:')
            self.pretty_print_transition_matrix(empirical_transition_matrix)
            print('\nInitialized transition matrix:')
            self.pretty_print_transition_matrix(self.transition_matrix.data)

    def pretty_print_transition_matrix(self, transition_matrix, tag_seq_indexer=None):
        if tag_seq_indexer is None:
            tag_seq_indexer = self.tag_seq_indexer
        str = '%10s' % ''
        for i in range(tag_seq_indexer.get_items_count()):
            str += '%10s' % tag_seq_indexer.idx2item_dict[i]
        str += '\n'
        for i in range(tag_seq_indexer.get_items_count()):
            str += '\n%10s' % tag_seq_indexer.idx2item_dict[i]
            for j in range(tag_seq_indexer.get_items_count()):
                str += '%10s' % ('%1.1f' % transition_matrix[i, j])
        print(str)

    def is_cuda(self):
        return self.transition_matrix.is_cuda

    def numerator(self, features_rnn_compressed, states_tensor, mask_tensor):
        # features_input_tensor: batch_num x max_seq_len x states_num
        # states_tensor: batch_num x max_seq_len
        # mask_tensor: batch_num x max_seq_len
        batch_num, max_seq_len = mask_tensor.shape
        score = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
        start_states_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, 1, dtype=torch.long).fill_(self.sos_idx))
        states_tensor = torch.cat([start_states_tensor, states_tensor], 1)
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:, n]
            curr_emission = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            curr_transition = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            for k in range(batch_num):
                curr_emission[k] = features_rnn_compressed[k, n, states_tensor[k, n + 1]].unsqueeze(0)
                curr_states_seq = states_tensor[k]
                curr_transition[k] = self.transition_matrix[curr_states_seq[n + 1], curr_states_seq[n]].unsqueeze(0)
            score = score + curr_emission*curr_mask + curr_transition*curr_mask
        return score

    def denominator(self, features_rnn_compressed, mask_tensor):
        # features_rnn_compressed: batch x max_seq_len x states_num
        # mask_tensor: batch_num x max_seq_len
        batch_num, max_seq_len = mask_tensor.shape
        score = self.tensor_ensure_gpu(torch.zeros(batch_num, self.states_num, dtype=torch.float).fill_(-9999.0))
        score[:, self.sos_idx] = 0.
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:, n].unsqueeze(-1).expand_as(score)
            curr_score = score.unsqueeze(1).expand(-1, *self.transition_matrix.size())
            curr_emission = features_rnn_compressed[:, n].unsqueeze(-1).expand_as(curr_score)
            curr_transition = self.transition_matrix.unsqueeze(0).expand_as(curr_score)
            #curr_score = torch.logsumexp(curr_score + curr_emission + curr_transition, dim=2)
            curr_score = log_sum_exp(curr_score + curr_emission + curr_transition)
            score = curr_score * curr_mask + score * (1 - curr_mask)
        #score = torch.logsumexp(score, dim=1)
        score = log_sum_exp(score)
        return score

    def decode_viterbi(self, features_rnn_compressed, mask_tensor):
        # features_rnn_compressed: batch x max_seq_len x states_num
        # mask_tensor: batch_num x max_seq_len
        batch_size, max_seq_len = mask_tensor.shape
        seq_len_list = [int(mask_tensor[k].sum().item()) for k in range(batch_size)]
        # Step 1. Calculate scores & backpointers
        score = self.tensor_ensure_gpu(torch.Tensor(batch_size, self.states_num).fill_(-9999.))
        score[:, self.sos_idx] = 0.0
        backpointers = self.tensor_ensure_gpu(torch.LongTensor(batch_size, max_seq_len, self.states_num))
        for n in range(max_seq_len):
            curr_emissions = features_rnn_compressed[:, n]
            curr_score = self.tensor_ensure_gpu(torch.Tensor(batch_size, self.states_num))
            curr_backpointers = self.tensor_ensure_gpu(torch.LongTensor(batch_size, self.states_num))
            for curr_state in range(self.states_num):
                T = self.transition_matrix[curr_state, :].unsqueeze(0).expand(batch_size, self.states_num)
                max_values, max_indices = torch.max(score + T, 1)
                curr_score[:, curr_state] = max_values
                curr_backpointers[:, curr_state] = max_indices
            curr_mask = mask_tensor[:, n].unsqueeze(1).expand(batch_size, self.states_num)
            score = score * (1 - curr_mask) + (curr_score + curr_emissions) * curr_mask
            backpointers[:, n, :] = curr_backpointers # shape: batch_size x max_seq_len x state_num
        best_score_batch, last_best_state_batch = torch.max(score, 1)
        # Step 2. Find the best path
        best_path_batch = [[state] for state in last_best_state_batch.tolist()]
        for k in range(batch_size):
            curr_best_state = last_best_state_batch[k]
            curr_seq_len = seq_len_list[k]
            for n in reversed(range(1, curr_seq_len)):
                curr_best_state = backpointers[k, n, curr_best_state].item()
                best_path_batch[k].insert(0, curr_best_state)
        return best_path_batch
#
# PAD_IDX = 0
# SOS_IDX = 1
# EOS_IDX = 2
# # UNK_IDX = 3
# class LayerCRF(nn.Module):
#     # def __init__(self, num_tags):
#     def __init__(self,gpu, states_num, pad_idx, sos_idx, tag_seq_indexer, verbose=True):
#         super(LayerCRF, self).__init__()
#         self.batch_size = 0
#         self.num_tags = states_num
#
#         # matrix of transition scores from j to i
#         self.trans = nn.Parameter(torch.randn([states_num, states_num]))
#         self.trans.data[SOS_IDX, :] = -10000 # no transition to SOS
#         self.trans.data[:, EOS_IDX] = -10000 # no transition from EOS except to PAD
#         self.trans.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
#         self.trans.data[PAD_IDX, :] = -10000 # no transition to PAD except from EOS
#         self.trans.data[PAD_IDX, EOS_IDX] = 0
#         self.trans.data[PAD_IDX, PAD_IDX] = 0
#
#     def forward(self, h, mask): # forward algorithm
#         # initialize forward variables in log space
#         score = Tensor(self.batch_size, self.num_tags).fill_(-10000) # [B, C]
#         score[:, SOS_IDX] = 0.
#         trans = self.trans.unsqueeze(0) # [1, C, C]
#         for t in range(h.size(1)): # recursion through the sequence
#             mask_t = mask[:, t].unsqueeze(1)
#             emit_t = h[:, t].unsqueeze(2) # [B, C, 1]
#             score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
#             score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
#             score = score_t * mask_t + score * (1 - mask_t)
#         score = log_sum_exp(score + self.trans[EOS_IDX])
#         return score # partition function
#
#     def score(self, h, y0, mask): # calculate the score of a given sequence
#         score = Tensor(self.batch_size).fill_(0.)
#         h = h.unsqueeze(3)
#         trans = self.trans.unsqueeze(2)
#         for t in range(h.size(1)): # recursion through the sequence
#             mask_t = mask[:, t]
#             emit_t = torch.cat([h[t, y0[t + 1]] for h, y0 in zip(h, y0)])
#             trans_t = torch.cat([trans[y0[t + 1], y0[t]] for y0 in y0])
#             score += (emit_t + trans_t) * mask_t
#         last_tag = y0.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
#         score += self.trans[EOS_IDX, last_tag]
#         return score
#
#     def decode_viterbi(self, h, mask): # Viterbi decoding
#         # initialize backpointers and viterbi variables in log space
#         bptr = torch.LongTensor()
#         score = torch.Tensor(self.batch_size, self.num_tags).fill_(-9999.0)
#
#         print('score 1: ', score)
#         score[:, SOS_IDX] = 0.
#         print('score',score)
#         for t in range(h.size(1)): # recursion through the sequence
#             mask_t = mask[:, t].unsqueeze(1)
#             print('score.unsqueeze(1)', score.unsqueeze(1))
#             print('self.trans', self.trans)
#             score_t = self.trans+ score.unsqueeze(1).cuda()  # [B, 1, C] -> [B, C, C]
#             print('score_t 1111',score_t)
#             # if len(score_t)== 0:
#             #     score_t = torch.randn([self.num_tags, self.num_tags])
#             # else:
#             score_t, bptr_t = score_t.max(2) # best previous scores and tags
#             score_t += h[:, t] # plus emission scores
#             bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
#             score = score_t * mask_t + score * (1 - mask_t)
#         score += self.trans[EOS_IDX].cpu()
#         best_score, best_tag = torch.max(score, 1)
#
#         # back-tracking
#         bptr = bptr.tolist()
#         best_path = [[i] for i in best_tag.tolist()]
#         for b in range(self.batch_size):
#             i = best_tag[b] # best tag
#             j = int(mask[b].sum().item())
#             for bptr_t in reversed(bptr[b][:j]):
#                 i = bptr_t[i]
#                 best_path[b].append(i)
#             best_path[b].pop()
#             best_path[b].reverse()
#
#         return best_path