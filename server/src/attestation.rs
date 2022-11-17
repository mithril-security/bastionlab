use uapi::_IOWR;
use nix::sys::ioctl;
use std::fs::File;

struct sev_snp_guest_request {
    uint8_t req_msg_type,
    uint8_t rsp_msg_type,
    uint8_t msg_version,
    uint16_t request_len,
    uint64_t request_uaddr,
    uint16_t response_len,
    uint64_t response_uaddr,
    uint32_t error,         /* firmware error code on failure (see psp-sev.h) */
}


enum snp_msg_type {
    SNP_MSG_TYPE_INVALID = 0,
    SNP_MSG_CPUID_REQ,
    SNP_MSG_CPUID_RSP,
    SNP_MSG_KEY_REQ,
    SNP_MSG_KEY_RSP,
    SNP_MSG_REPORT_REQ,
    SNP_MSG_REPORT_RSP,
    SNP_MSG_EXPORT_REQ,
    SNP_MSG_EXPORT_RSP,
    SNP_MSG_IMPORT_REQ,
    SNP_MSG_IMPORT_RSP,
    SNP_MSG_ABSORB_REQ,
    SNP_MSG_ABSORB_RSP,
    SNP_MSG_VMRK_REQ,
    SNP_MSG_VMRK_RSP,
    SNP_MSG_TYPE_MAX
}

macro_rules! SEV_GUEST_IOC_TYPE {()=>{'S'};}
macro_rules! SEV_SNP_GUEST_MSG_REQUEST {() => {_IOWR(SEV_GUEST_IOC_TYPE, 0x0, struct sev_snp_guest_request)};}
macro_rules! SEV_SNP_GUEST_MSG_REPORT {() => {_IOWR(SEV_GUEST_IOC_TYPE, 0x1, struct sev_snp_guest_request)};}
macro_rules! SEV_SNP_GUEST_MSG_KEY {() => {_IOWR(SEV_GUEST_IOC_TYPE, 0x2, struct sev_snp_guest_request)};}


fn main() {

    uint8_t report_in[96];
    uint8_t report_out[1280];
    int fd;
    int i;
    int rc;
    //FILE* freport;

    let args: Vec<String> = env::args().collect();

    struct sev_snp_guest_request payload = {
        .req_msg_type = SNP_MSG_REPORT_REQ,
        .rsp_msg_type = SNP_MSG_REPORT_RSP,
        .msg_version = 1,
        .request_len = sizeof(report_in),
        .request_uaddr = (uint64_t)report_in,
        .response_len = sizeof(report_out),
        .response_uaddr = (uint64_t)report_out,
        .error = 0
    };

    //memset(report_in, 0, sizeof(report_in));
    //memset(report_out, 0, sizeof(report_out)); 

    char data[15] = "abcdefghijklmno";
    for index in 0..15 {     //(int x=0; x<15; x++)
        report_in[index] = data[index];     //ASCII values string to int
    }

    if(args.len() != 2) {
        println!("Usage: getreport <filename>\n");
        return;
    }

    fd = File::open("/dev/sev", O_RDWR | O_CLOEXEC);

    if (fd < 0) {
        println!("Failed to open /dev/sev\n");
        return;
    }

    rc = ioctl_read!(fd, SEV_SNP_GUEST_MSG_REPORT, &payload);

    if (rc != 0) {
        println!("ioctl failed (%x)\n", rc);
        return;
    }

    for (i = 0; i < payload.response_len; i++) {
        println!("{:#02x}", report_out[i]);
        if (i % 16 == 15)
            println!("\n");
        else
            println!(" ");
    }

    let mut freport = File::create(args[1]);    //, "wb");

    if(!freport) {
        println!("Could not create output report file: %s\n", argv[1]);
        return;
    }

    File::write(report_out+32, 1, 1184, freport);

    //fclose(freport);

    println!("\nReport generated: %s\n", args[1]);

}