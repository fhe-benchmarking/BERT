import pickle
import sys

import lattica_query.query_toolkit as toolkit_interface
from lattica_query.serialization.hom_op_pb2 import (
    QueryClientSequentialHomOp as ProtoQueryClientSequentialHomOp,
)

import submission_utils
local_file_paths = submission_utils.init(int(sys.argv[1]))

# Read data from local filesystem required for encoding and encrypting
with open(local_file_paths.PT_PATH, "rb") as f:
    pt_ser = f.read()

context = pickle.load(open(local_file_paths.PATH_CONTEXT, "rb"))
hom_seq = pickle.load(open(local_file_paths.PATH_HOM_SEQ, "rb"))
sk =      pickle.load(open(local_file_paths.PATH_SK, "rb"))

homsec_proto = ProtoQueryClientSequentialHomOp()
homsec_proto.ParseFromString(hom_seq)
block_proto = homsec_proto.client_blocks[0]
pt_axis_external = block_proto.pt_axis_external if block_proto.HasField("pt_axis_external") else None

pt_enc = toolkit_interface.apply_client_block(block_proto.SerializeToString(), context, pt_ser)
ct_batch = toolkit_interface.enc(context, sk, pt_enc, pack_for_transmission=True, n_axis_external=pt_axis_external)
pickle.dump(ct_batch, open(local_file_paths.get_ct_upload_path("batch"), "wb"))
