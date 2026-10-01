use std::fmt;

/// Generic trait for error messages in Tantale.
pub trait ErrMsg {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result;
}

#[macro_export]
macro_rules! err_msg {
    ($type: ty) => {
        impl std::error::Error for $type {}
        impl std::fmt::Display for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self._err_msg(f)
            }
        }
        impl std::fmt::Debug for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self._err_msg(f)
            }
        }
    };
}


/// Error type for checkpointing operations from a [`Checkpointer`](crate::Checkpointer).
pub enum CheckpointerError {
    WriteError(String),
    ReadError(String),
    FolderCreationError(String),
    FileCreationError(String),
    FileNotFound(String),
    Other(String),
}

impl ErrMsg for CheckpointerError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CheckpointerError::WriteError(err) => {
                write!(f, "Checkpointer write error: {}", err)
            }
            CheckpointerError::ReadError(err) => {
                write!(f, "Checkpointer read error: {}", err)
            }
            CheckpointerError::FolderCreationError(err) => {
                write!(f, "Checkpointer folder creation error: {}", err)
            }
            CheckpointerError::FileCreationError(err) => {
                write!(f, "Checkpointer file creation error: {}", err)
            }
            CheckpointerError::FileNotFound(err) => {
                write!(f, "Checkpointer file not found error: {}", err)
            }
            CheckpointerError::Other(msg) => {
                write!(f, "Checkpointer error: {}", msg)
            }
        }
    }
}
err_msg!(CheckpointerError);

/// Error type for checkpointing operations from a [`Checkpointer`](crate::Checkpointer).
pub enum CommunicatorError {
    /// Error during sending a message to a peer.
    SendError(String),
    /// Error during receiving a message from a peer.
    ReceiveError(String),
    /// Error during serialization of a message.
    SerializeError(String),
    /// Error during deserialization of a message.
    DeserializeError(String),
    /// Error when the communication channel is closed.
    ChannelClosed,
    /// Error when the received message is of an unexpected type.
    InvalidMessage,
}
impl ErrMsg for CommunicatorError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CommunicatorError::SendError(err) => {
                write!(f, "Channel send error: {}", err)
            }
            CommunicatorError::ReceiveError(err) => {
                write!(f, "Channel receive error: {}", err)
            }
            CommunicatorError::SerializeError(err) => {
                write!(f, "Channel serialize error: {}", err)
            }
            CommunicatorError::DeserializeError(err) => {
                write!(f, "Channel deserialize error: {}", err)
            }
            CommunicatorError::ChannelClosed => {
                write!(f, "Channel closed")
            }
            CommunicatorError::InvalidMessage => {
                write!(f, "Invalid message type")
            }
        }
    }
}
err_msg!(CommunicatorError);

/// Error type for [`Onto`](crate::domain::onto::Onto) operations between domains.
pub struct OntoError(pub String);

impl ErrMsg for OntoError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = "Onto error";
        write!(f, "{}, {}.", err_msg, self.0)
    }
}
err_msg!(OntoError);

/// Error type for checkpointing operations from a [`Recorder`](crate::Recorder).
pub enum RecorderError {
    WriteError(String),
    FolderCreationError(String),
    FileCreationError(String),
    FileNotFound(String),
    Other(String),
}

impl ErrMsg for RecorderError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RecorderError::WriteError(err) => {
                write!(f, "Recorder write error: {}", err)
            }
            RecorderError::FolderCreationError(err) => {
                write!(f, "Recorder folder creation error: {}", err)
            }
            RecorderError::FileCreationError(err) => {
                write!(f, "Recorder file creation error: {}", err)
            }
            RecorderError::FileNotFound(err) => {
                write!(f, "Recorder file not found error: {}", err)
            }
            RecorderError::Other(msg) => {
                write!(f, "Recorder error: {}", msg)
            }
        }
    }
}
err_msg!(RecorderError);

/// Error type for evaluation operations from an [`Evaluator`](crate::Evaluator).
pub enum EvaluationError {
    NotEvaluated(String),
}
impl ErrMsg for EvaluationError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvaluationError::NotEvaluated(err) => {
                write!(f, "Evaluation error: {}", err)
            }
        }
    }
}
err_msg!(EvaluationError);