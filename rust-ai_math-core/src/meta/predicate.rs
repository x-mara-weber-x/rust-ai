/// Allows restriction of generic const parameters in a `where` clause.
pub enum Predicate<const EXPRESSION: bool> {}

pub trait Satisfied {}
impl Satisfied for Predicate<true> {}
