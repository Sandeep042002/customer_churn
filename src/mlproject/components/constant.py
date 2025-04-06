import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pydantic import BaseModel, Field
from typing import Optional
import logging
from components.utils import get_logger
import pandas as pd
from enum import Enum
import datetime


logger = get_logger(__name__)


class Job(str, Enum):
    ADMIN = "admin."
    BLUE_COLLAR = "blue-collar"
    ENTREPRENEUR = "entrepreneur"
    HOUSEMAID = "housemaid"
    MANAGEMENT = "management"
    RETIRED = "retired"
    SELF_EMPLOYED = "self-employed"
    SERVICES = "services"
    STUDENT = "student"
    TECHNICIAN = "technician"
    UNEMPLOYED = "unemployed"
    UNKNOWN = "unknown"


class MaritalStatus(str, Enum):
    DIVORCED = "divorced"
    MARRIED = "married"
    SINGLE = "single"


class Education(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    UNKNOWN = "unknown"


class Month(str, Enum):
    JAN = "jan"
    FEB = "feb"
    MAR = "mar"
    APR = "apr"
    MAY = "may"
    JUN = "jun"
    JUL = "jul"
    AUG = "aug"
    SEP = "sep"
    OCT = "oct"
    NOV = "nov"
    DEC = "dec"


class PrevOutcome(str, Enum):
    FAILURE = "failure"
    OTHER = "other"
    SUCCESS = "success"
    UNKNOWN = "unknown"


class CallType(str, Enum):
    UNKNOWN = "unknown"
    CELLULAR = "cellular"
    TELEPHONE = "telephone"


class PredictionRequest(BaseModel):
    conda_age: float = Field(alias="conda age", gt=0, description="Customer age")
    job: Job  # Assuming jobs are variable
    marital: MaritalStatus
    education_qual: Education
    day: int = Field(..., ge=1, le=31, description="Day of the month")
    call_type: CallType
    mon: Month
    dur: float = Field(..., gt=0, description="Call duration in seconds")
    num_calls: int = Field(..., ge=0, description="Number of calls")
    prev_outcome: PrevOutcome

    class Config:
        allow_population_by_field_name = True
        # extra = "forbid"


# if __name__ == "__main__":
#     sample_data = {
#         "conda age": [25, -5, 30],  # One invalid age (-5)
#         "dur": [120, 0, 300],  # One invalid duration (0)
#         "num_calls": [3, 2, -1],  # One invalid num_calls (-1)
#         "job": ["admin", "technician", "blue-collar"],
#         "marital": ["married", "single", "divorced"],
#         "education_qual": ["secondary", "tertiary", "primary"],
#         "call_type": ["cellular", "telephone", "cellular"],
#         "mon": ["jan", "feb", "mar"],
#         "prev_outcome": ["failure", "success", None],
#         "y": ["yes", "no", "yes"],
#     }
#     df = pd.DataFrame(sample_data)

#     validated_df = CustomerData.validate_dataframe(df)
#     print(validated_df)
