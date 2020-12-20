
#[cfg(feature = "gpu")]
#[test]
pub fn test_pure_gpu_fft() {
    use std::time::{Duration, Instant};
    use bellperson::gpu::*;
    use bellperson::groth16::tests::dummy_engine::*;

    let now = Instant::now();
    let mut kernel:FFTKernel<DummyEngine> = FFTKernel::create(true);
    let mut engine = DummyEngine;
    let mut a = [Fr::zero();32];
    kernel.radix_fft(&mut a,&Fr::zero(),4);
    println!("{}",a);
    println!(
        "Lower proof gen finished in {}s and {}ms",
        now.elapsed().as_secs(),
        now.elapsed().subsec_nanos() / 1000000
    );
}