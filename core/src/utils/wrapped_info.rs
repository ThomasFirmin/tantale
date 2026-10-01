use serde::{Deserialize, Serialize};

use crate::{SolInfo, utils::record::{Recordable, Schema}};

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

impl<Info, SuperInfo> Schema<()> for WrappedInfo<Info, SuperInfo>
where
    Info: SolInfo,
    SuperInfo: SolInfo,
{
    type Parts = ();
    fn columns(context: &(), part: ()) -> Vec<String> {
        let mut schema = SuperInfo::columns(context, part);
        schema.extend(Info::columns(context, part));
        schema
    }
}

impl<Info, SuperInfo> Recordable<()> for WrappedInfo<Info, SuperInfo>
where
    Info: SolInfo,
    SuperInfo: SolInfo,
{
    type Parts = ();

    fn record(&self, context: &(), part: Self::Parts) -> Vec<String> {
        let mut record = self.super_info.record(context, part);
        record.extend(self.inner.record(context, part));
        record
    }
}

impl<Info, SuperInfo> SolInfo for WrappedInfo<Info, SuperInfo>
where
    Info: SolInfo,
    SuperInfo: SolInfo,
{
}