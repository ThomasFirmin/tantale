
use serde::{Deserialize, Serialize};
use std::{any::{Any, TypeId}, sync::Arc};

use crate::{SolInfo, utils::record::{Recordable, Schema}};

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "Info: SolInfo, SuperInfo: SolInfo")]
pub struct WrappedInfo<Info, SuperInfo>
where
    Info: SolInfo,
    SuperInfo: SolInfo,
{
    pub inner: Arc<Info>,
    pub super_info: Arc<SuperInfo>,
}

impl<Info, SuperInfo> WrappedInfo<Info, SuperInfo>
where
    Info: SolInfo,
    SuperInfo: SolInfo,
{
    pub fn new(inner: Arc<Info>, super_info: Arc<SuperInfo>) -> Self {
        WrappedInfo { inner, super_info }
    }
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
    fn find_info<T: SolInfo>(self: Arc<Self>) -> Option<Arc<T>> {
        if TypeId::of::<Self>() == TypeId::of::<T>() {
            return (self as Arc<dyn Any + Send + Sync>).downcast::<T>().ok();
        }
        self.inner.clone().find_info::<T>()
            .or_else(|| self.super_info.clone().find_info::<T>())
    }
}