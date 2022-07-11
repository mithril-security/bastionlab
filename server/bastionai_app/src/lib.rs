use tch::*;

pub mod file_system {
    pub struct UserMode {}
    pub struct Mode {}

    pub fn store_artifacts<'a, T>(
        artifacts: Vec<T>,
        path: &'a str,
        user_mode: UserMode,
        chunk_size: usize,
        mode: Mode,
        serialization_fn: impl Fn(Vec<T>, String) -> T,
        single_mode: bool,
    ) -> Result<(), &'a str>
    where
        T: Clone + Copy,
    {
        let mut batch = Vec::new();
        let mut index: usize = 0;
        let mut length: usize = 0;

        // std::fs::create_dir(path.clone()).unwrap();

        // /* Spot for torch.save() */
        // TrainableCModule

        if single_mode {
            let artifacts_list: Vec<T> = artifacts.clone().into_iter().map(|x| x).collect();

            if artifacts_list.len() != 1 {
                return Err("Error");
            }

            serialization_fn(
                vec![artifacts.clone()[0]],
                format!("{}/object.pt", path.clone()),
            );
        } else {
            for artifact in artifacts {
                length += 1;
                if batch.len() < chunk_size {
                    batch.push(artifact);
                } else {
                    serialization_fn(batch, format!("{}/chunk_{}.pt", path.clone(), index));
                    index += 1;

                    batch = vec![artifact];
                }
            }

            serialization_fn(batch, format!("{}/chunk_{}.pt", path.clone(), index));

            /* Spot for torch.save() */
        }
        Ok(())
    }

    // pub fn unstream_artifacts<T>(stream: Vec<u8>, impl Fn(T)-)
}
