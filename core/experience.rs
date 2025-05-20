pub trait Experiment{
    fn get_process_id() -> u32;
}

pub trait MPIExp : Experiment{
    fn get_process_id()->u32;
}

pub trait RegExp : Experiment{
    fn get_process_id()->u32{
        std::process::id();
    }
}