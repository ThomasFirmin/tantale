use crate::optimizer::OptState;
pub trait Stop<State>
where
    State:OptState
{
    fn stop(&self) -> bool;
    fn update(&mut self,
        state_opt : State,
    );
}