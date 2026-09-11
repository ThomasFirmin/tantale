use serde::{Deserialize, Serialize};

use crate::{SolInfo, Tabular, utils::has_trait::HasHeader};

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "Info: SolInfo, SuperInfo: SolInfo")]
pub struct WrappedInfo<Info, SuperInfo>
where
    Info: SolInfo,
    SuperInfo: SolInfo,
{
    pub inner: Info,
    pub super_info: SuperInfo,
}

impl<Info, SuperInfo> HasHeader for WrappedInfo<Info, SuperInfo>
where
    Info: SolInfo,
    SuperInfo: SolInfo,
{
    
    fn header() -> Vec<String> {
        let mut sinfo = SuperInfo::header();
        let mut info = Info::header();
        sinfo.append(&mut info);
        sinfo
    }
}

impl<Info, SuperInfo> Tabular<()> for WrappedInfo<Info, SuperInfo>
where
    Info: SolInfo + Tabular<(), Parts = ()>,
    SuperInfo: SolInfo + Tabular<(), Parts = ()>,
{
    type Parts = ();
    fn column_names(&self, part: Self::Parts) -> Vec<String> {
        Self::header()
    }

    fn column_values(&self, elem: &(), part: Self::Parts) -> Vec<String> {
        let mut info_write = self.inner.column_values(&(), ());
        let super_write = self.super_info.column_values(&(), ());
        info_write.extend(super_write);
        info_write
    }
    
}

impl<Info, SuperInfo> SolInfo for WrappedInfo<Info, SuperInfo>
where
    Info: SolInfo,
    SuperInfo: SolInfo,
{
}