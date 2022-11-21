use std::ops::{BitOr, BitAnd, BitOrAssign, BitAndAssign};

#[derive(Debug, Clone)]
pub enum Access {
    Granted,
    Denied(String),
}

impl BitOr for Access {
    type Output = Self;

    fn bitor(mut self, rhs: Self) -> Self::Output {
        self |= rhs;
        self
    }
}

impl BitOrAssign for Access {
    fn bitor_assign(&mut self, rhs: Self) {
        match self {
            Access::Granted => (),
            Access::Denied(_) => *self = rhs,
        }
    }
}

impl BitAnd for Access {
    type Output = Self;
    
    fn bitand(mut self, rhs: Self) -> Self::Output {
        self &= rhs;
        self
    }
}

impl BitAndAssign for Access {
    fn bitand_assign(&mut self, rhs: Self) {
        match self {
            Access::Granted => *self = rhs,
            Access::Denied(_) => (),
        }
    }
}
