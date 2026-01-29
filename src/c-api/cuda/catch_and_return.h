#pragma once

#define CATCH_AND_RETURN_ERRORCODE                                                                                     \
    catch (const MPPException &mppex)                                                                                  \
    {                                                                                                                  \
        ErrorMessageSingleton::SetLastError(mppex.what(), mppex.GetCode());                                            \
        return static_cast<MPPErrorCode>(mppex.GetCode());                                                             \
    }                                                                                                                  \
    catch (const std::exception &ex)                                                                                   \
    {                                                                                                                  \
        ErrorMessageSingleton::SetLastError(ex.what(), ExceptionCode::Unknown);                                        \
        return MPPErrorCode::MPP_ERROR_UNKNOWN;                                                                        \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {                                                                                                                  \
        return MPPErrorCode::MPP_ERROR_UNKNOWN;                                                                        \
    }                                                                                                                  \
    return MPP_SUCCESS;
