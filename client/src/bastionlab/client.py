from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING, Optional
import grpc
from bastionlab.pb.bastionlab_pb2 import (
    ReferenceRequest,
    ReferenceResponse,
    Query,
    Empty,
)
from bastionlab.pb.bastionlab_pb2_grpc import BastionLabStub
import polars as pl

from bastionlab.utils import (
    deserialize_dataframe,
    serialize_dataframe,
)

if TYPE_CHECKING:
    from bastionlab.remote_polars import RemoteLazyFrame, FetchableLazyFrame


#<!-- Attestation dependencies -->
from bastionai.pb.attestation_pb2 import ReportRequest, ReportResponse
from bastionai.pb.attestation_pb2_grpc import AttestationStub
import base64
import _attestation_c


class Client:
    def __init__(self, stub: BastionLabStub):
        self.stub = stub

    def send_df(self, df: pl.DataFrame) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.SendDataFrame(serialize_dataframe(df))
        return FetchableLazyFrame._from_reference(self, res)

    def _fetch_df(self, ref: List[str]) -> pl.DataFrame:
        joined_bytes = b""
        for b in self.stub.FetchDataFrame(ReferenceRequest(identifier=ref)):
            joined_bytes += b.data

        return deserialize_dataframe(joined_bytes)

    def _run_query(
        self,
        composite_plan: str,
    ) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.RunQuery(Query(composite_plan=composite_plan))
        return FetchableLazyFrame._from_reference(self, res)

    def list_dfs(self) -> List["FetchableLazyFrame"]:
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.ListDataFrames(Empty()).list
        return [FetchableLazyFrame._from_reference(self, ref) for ref in res]

    def get_df(self, identifier: str) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.GetDataFrameHeader(ReferenceRequest(identifier=identifier))
        return FetchableLazyFrame._from_reference(self, res)


def get_validate_attestation(attestation_client: Client, server_cert: str):
    import secrets
    import requests
    import hashlib

    nonce = secrets.token_bytes(16)
    nonce = base64.b64encode(nonce)
    
    report_response = GRPCException.map_error(lambda: attestation_client.stub.ClientReportRequest(ReportRequest(nonce=nonce)))
    report = report_response.report

    hasher = hashlib.sha256()
    hasher.update(nonce+server_cert.encode('utf-8'))
    calc_measurement = hasher.digest()
 
    cert_start_line = '-----BEGIN CERTIFICATE-----'
    
    cert_chain = requests.get("https://kdsintf.amd.com/vcek/v1/Milan/cert_chain")
    certs = cert_chain.text.split(cert_start_line)
    
    #First ASK then ARK according to the AMD specifications
    ARK = cert_start_line+certs[2]
    ASK = cert_start_line+certs[1]

    #Comparison of expected MRENCLAVE against received MRENCLAVE
    #This value is obtained when the UEFI image is generated
    """
    MRENCLACE="XXXXXXXXXXXXXXXXXXXXX"
    if MRENCLAVE == report[48:80]:
        print("MRENCLAVE is expected value")
    else:
        print("MRENCLAVE does not match expected value. Terminating ...")
        exit()
    """
    
    #This should cover the entire user-supplied data range (512 bits)
    #currently it only compares the exact size of the hash 256 bits.
    if calc_measurement != report[112:144]:     
        print("Nonce and server certification validation failed. Terminating connection...")
        exit()
    else:
        print("Nonce and server certificate validated successfully")

    vcek_pem = ssl.DER_cert_to_PEM_cert(report_response.vcek_cert)
    
    ret_val = 0
    ret_val = _attestation_c.attest(report[32:1216],vcek_pem,ASK,ARK,len(vcek_pem),len(ASK),len(ARK))
    
    if ret_val != 0:
        print("Attestation validation failed. Terminating connection...")
        exit()

@dataclass
class Connection:
    host: str
    port: int
    channel: Any = None
    server_name: str = "bastionlab-srv"
    _client: Optional[Client] = None

    @property
    def client(self) -> Client:
        if self._client is not None:
            return self._client
        else:
            return self.__enter__()

    def close(self):
        if self._client is not None:
            self.__exit__(None, None, None)

    ###Recheck how the tls tunnel is implemented, a basic implementation is used here mainly to retrieve the server cert to
    ###compare against the hash in the attestation report
    def __enter__(self) -> Client:
        #server_target = f"{self.host}:{self.port}"
        #self.channel = grpc.insecure_channel(server_target)

        connection_options = (("grpc.ssl_target_name_override", self.server_name),)
        server_cert = ssl.get_server_certificate((self.host, self.port))

        server_cred = grpc.ssl_channel_credentials(
            root_certificates=bytes(server_cert, encoding="utf8")
        )

        server_target = f"{self.host}:{self.port}"
        self.channel = grpc.secure_channel(
            server_target, server_cred, options=connection_options
        )

        if os.environ['ATTESTATION'] == "true":
            get_validate_attestation(Client(AttestationStub(self.channel)), server_cert)

        self._client = Client(BastionLabStub(self.channel))
        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self._client = None
        self.channel.close()
