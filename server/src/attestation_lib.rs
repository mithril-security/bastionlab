use nix::ioctl_readwrite;
use std::os::unix::io::AsRawFd;
use std::fs::OpenOptions;
use std::collections::HashMap;
use regex::Regex;

#[repr(C)]
pub struct sev_snp_guest_request {
    req_msg_type: u8,
    rsp_msg_type: u8,
    msg_version: u8,
    request_len: u16,
    request_uaddr: *const u8,
    response_len: u16,
    response_uaddr: *const u8,
    error: u32,         /* firmware error code on failure (see psp-sev.h) */
}

#[repr(C)]
enum snp_msg_type {
    SNP_MSG_TYPE_INVALID = 0,
    SNP_MSG_CPUID_REQ = 1,
    SNP_MSG_CPUID_RSP = 2,
    SNP_MSG_KEY_REQ = 3,
    SNP_MSG_KEY_RSP = 4,
    SNP_MSG_REPORT_REQ = 5,
    SNP_MSG_REPORT_RSP = 6,
    SNP_MSG_EXPORT_REQ = 7,
    SNP_MSG_EXPORT_RSP,
    SNP_MSG_IMPORT_REQ,
    SNP_MSG_IMPORT_RSP,
    SNP_MSG_ABSORB_REQ,
    SNP_MSG_ABSORB_RSP,
    SNP_MSG_VMRK_REQ,
    SNP_MSG_VMRK_RSP,
    SNP_MSG_TYPE_MAX
}

const SEV_GUEST_IOC_TYPE:u8 = b'S';

ioctl_readwrite!(SEV_SNP_GUEST_MSG_REQUEST, SEV_GUEST_IOC_TYPE, 0x0, sev_snp_guest_request);
ioctl_readwrite!(SEV_SNP_GUEST_MSG_KEY, SEV_GUEST_IOC_TYPE, 0x2, sev_snp_guest_request);
ioctl_readwrite!(get_report_from_fw, SEV_GUEST_IOC_TYPE, 0x1, sev_snp_guest_request);


pub async fn get_report(input:Vec<u8>) -> Option<HashMap<String,Vec<u8>>>{

    let report_out = generate_report(input);

    let report_certs = validate_guest_attestation(report_out.unwrap());
    
    return Some(report_certs.await)
}

pub fn generate_report(input:Vec<u8>) -> Option<[u8;1280]>{
    let input_bytes:Vec<u8> = input;     //Can be upto 512 bits
    
    let mut report_in: [u8;96] = [0;96]; 
    
    let mut report_out: [u8;1280] = [0;1280];
    let mut i: usize = 0;
    
    for elem in input_bytes {
        report_in[i] = elem;
        i = i+1;
    }
      
    let mut payload = sev_snp_guest_request {
        req_msg_type : snp_msg_type::SNP_MSG_REPORT_REQ as u8,
        rsp_msg_type : snp_msg_type::SNP_MSG_REPORT_RSP as u8,
        msg_version : 1,
        request_len : 96,   //size specified in the azure script
        request_uaddr : report_in.as_ptr(),
        response_len : 1280,        //size specified in the azure script
        response_uaddr : report_out.as_ptr(),
        error : 0
    };

  
    unsafe {
      let fd = OpenOptions::new().read(true).write(true).open("/dev/sev");

        if fd.as_ref().unwrap().as_raw_fd() < 0 {
            println!("Failed to open /dev/sev\n");
            return None;
        }
  
      let rc = get_report_from_fw(fd.as_ref().unwrap().as_raw_fd(), &mut payload as *mut sev_snp_guest_request);
    
        if rc != Ok(0) {
            println!("ioctl failed {:?}", rc);
            return None;
        }
    }

    Some(report_out)
}

pub async fn validate_guest_attestation(report: [u8;1280]) ->  HashMap<String,Vec<u8>> {
    
    //HW id can be retrieved from the report and must be supplied in the url as hex
    let hwid = &report[448..512];
    
    //tcb list can be retrieved from the report (current tcb) and parsed and converted to decimal
    let re = Regex::new(r"\[|\]").unwrap();
    let blSPL_temp = &format!("{:?}",&report[416..417]);
    let blSPL =  re.replace_all(&blSPL_temp,"").to_string();
    let teeSPL_temp = &format!("{:?}",&report[417..418]);
    let teeSPL = re.replace_all(&teeSPL_temp,"").to_string();
    let snpSPL_temp = &format!("{:?}",&report[422..423]);
    let snpSPL = re.replace_all(&snpSPL_temp,"").to_string();
    let ucodeSPL_temp = &format!("{:?}",&report[423..424]);
    let ucodeSPL = re.replace_all(&ucodeSPL_temp,"").to_string();

    let hex_hwid = &hex::encode(&hwid);
    //Retrieves vcek cert from amd server
    let vcek_url = format!("{}{}{}{}{}{}{}{}{}{}","https://kdsintf.amd.com/vcek/v1/Milan/".to_owned(),hex_hwid,"?blSPL=",&blSPL,"&teeSPL=",&teeSPL,"&snpSPL=",&snpSPL,"&ucodeSPL=",&ucodeSPL);
    //let vcek_url = "https://kdsintf.amd.com/vcek/v1/Milan/".to_owned()+hex_hwid+"?blSPL="+&blSPL+"&teeSPL="+&teeSPL+"&snpSPL="+&snpSPL+"&ucodeSPL="+&ucodeSPL; 
    
    //Retrieves ask and ark from amd server
    let cert_chain_url = "https://kdsintf.amd.com/vcek/v1/Milan/cert_chain";
    
    //Can be retrieved from the reportm the value 1h means ECDSA p-384 Sha384 
    //but this seems to be the only available encoding according to the firmware specification.
    let signature_algo = "ECDSA_SHA384";
    let vcek_cert = reqwest::get(vcek_url).await.unwrap();
    let cert_chain = reqwest::get(cert_chain_url).await.unwrap();
    
    let mut report_certs = HashMap::new();
    report_certs.insert("signature_algo".to_string(),signature_algo.as_bytes().to_vec(),);
    report_certs.insert("vcek_cert".to_string(),vcek_cert.bytes().await.unwrap().to_vec(),);
    report_certs.insert("cert_chain".to_string(),cert_chain.bytes().await.unwrap().to_vec(),);
    report_certs.insert("report".to_string(),report.to_vec(),);

    return report_certs
}