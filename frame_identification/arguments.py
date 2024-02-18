import argparse

# class Args:
#     def __init__(self):
#         self.train = True
#         self.test = True
#         self.target_classifier = True 
#         self.gold_targets = True
#         self.candidate_targets = False
#         self.use_negatives = False
#         self.num_negatives = 5
#         self.use_lu_definitions = True
#         self.use_frame_definitions = True
#         self.clip_frame_definitions = True
#         self.append_fe_names = True
#         self.add_fsp_tokens = True
#         self.add_info_tokens = False
#         self.num_info_tokens = 16
#         self.run_name = None
#         self.model = None
#         self.epochs = 5
#         self.lr = 5e-5

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true', default=True)
parser.add_argument('--test', action='store_true', default=True)
parser.add_argument('--target_classifier', action='store_true', default=True)
parser.add_argument('--sequence_classifier', dest='target_classifier', action='store_false', default=True)
parser.add_argument('--gold_targets', action='store_true', default=True)
parser.add_argument('--candidate_targets', action='store_true', default=False)
parser.add_argument('--use_negatives', action='store_true', default=False)
parser.add_argument('--num_negatives', type=int, default=5)
parser.add_argument('--use_lu_definitions', action='store_true', default=True)
parser.add_argument('--no_lu_definitions', action='store_false', dest='use_lu_definitions')
parser.add_argument('--use_frame_definitions', action='store_true', default=True)
parser.add_argument('--no_frame_definitions', action='store_false', dest='use_frame_definitions')
parser.add_argument('--clip_frame_definitions', action='store_true', default=True)
parser.add_argument('--full_frame_definitions', action='store_false', dest='clip_frame_definitions')
parser.add_argument('--append_fe_names', action='store_true', default=True)
parser.add_argument('--no_append_fe_names', action='store_false', dest='append_fe_names')
parser.add_argument('--add_fsp_tokens', action='store_true', default=True)
parser.add_argument('--no_fsp_tokens', action='store_false', dest='add_fsp_tokens')
parser.add_argument('--add_info_tokens', action='store_true', default=False)
parser.add_argument('--num_info_tokens', type=int, default=16)
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=5e-5)

args = parser.parse_args()

# If using candidate targets, disable gold targets
if args.candidate_targets:
    args.gold_targets = False

print(args.use_lu_definitions)