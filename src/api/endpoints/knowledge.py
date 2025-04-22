"""
知识库API端点模块

提供知识库相关的API端点
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from src.config.settings import get_settings
from src.clients.dify_sdk_client import DifySDKAdapter

router = APIRouter()

class KnowledgeBaseResponse(BaseModel):
    """知识库响应模型"""
    message: str
    status: str

class MetadataCreate(BaseModel):
    """创建元数据请求模型"""
    type: str = Field(..., description="元数据类型")
    name: str = Field(..., description="元数据名称")

class MetadataUpdate(BaseModel):
    """更新元数据请求模型"""
    name: str = Field(..., description="元数据名称")

class MetadataResponse(BaseModel):
    """元数据响应模型"""
    id: str
    type: str
    name: str

class MetadataListResponse(BaseModel):
    """元数据列表响应模型"""
    metadata: List[MetadataResponse]

class DatasetCreate(BaseModel):
    """创建知识库请求模型"""
    name: str = Field(..., description="知识库名称")
    description: str = Field("", description="知识库描述")
    indexing_technique: str = Field("high_quality", description="索引技术")
    permission: Optional[Dict[str, Any]] = Field(None, description="权限设置")

class DatasetUpdate(BaseModel):
    """更新知识库请求模型"""
    name: Optional[str] = Field(None, description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    indexing_technique: Optional[str] = Field(None, description="索引技术")
    permission: Optional[Dict[str, Any]] = Field(None, description="权限设置")

class DatasetResponse(BaseModel):
    """知识库响应模型"""
    id: str
    name: str
    description: str
    indexing_technique: str
    permission: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str

@router.get("/", response_model=KnowledgeBaseResponse)
async def get_knowledge_base_status():
    """
    获取知识库状态
    
    Returns:
        KnowledgeBaseResponse: 知识库状态响应
    """
    return KnowledgeBaseResponse(
        message="知识库功能已可用",
        status="available"
    )

# 获取Dify客户端实例
def get_dify_client():
    """获取Dify客户端实例"""
    settings = get_settings()
    return DifySDKAdapter(
        api_key=settings.DIFY_API_KEY,
        api_base_url=settings.DIFY_API_BASE_URL
    )

@router.post("/datasets/{dataset_id}/metadata", response_model=MetadataResponse, status_code=status.HTTP_201_CREATED)
async def create_metadata(
    dataset_id: str = Path(..., description="知识库ID"),
    metadata: MetadataCreate = Field(...),
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    创建知识库元数据
    
    Args:
        dataset_id: 知识库ID
        metadata: 元数据创建请求
        
    Returns:
        MetadataResponse: 创建的元数据响应
    """
    try:
        result = dify_client.create_dataset_metadata(
            dataset_id=dataset_id,
            metadata_type=metadata.type,
            metadata_name=metadata.name
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建元数据失败: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/metadata", response_model=MetadataListResponse)
async def list_metadata(
    dataset_id: str = Path(..., description="知识库ID"),
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    获取知识库元数据列表
    
    Args:
        dataset_id: 知识库ID
        
    Returns:
        MetadataListResponse: 元数据列表响应
    """
    try:
        result = dify_client.get_dataset_metadata(dataset_id=dataset_id)
        return {"metadata": result.get("data", [])}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取元数据列表失败: {str(e)}"
        )

@router.patch("/datasets/{dataset_id}/metadata/{metadata_id}", response_model=MetadataResponse)
async def update_metadata(
    dataset_id: str = Path(..., description="知识库ID"),
    metadata_id: str = Path(..., description="元数据ID"),
    metadata: MetadataUpdate = Field(...),
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    更新知识库元数据
    
    Args:
        dataset_id: 知识库ID
        metadata_id: 元数据ID
        metadata: 元数据更新请求
        
    Returns:
        MetadataResponse: 更新后的元数据响应
    """
    try:
        result = dify_client.update_dataset_metadata(
            dataset_id=dataset_id,
            metadata_id=metadata_id,
            metadata_name=metadata.name
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新元数据失败: {str(e)}"
        )

@router.delete("/datasets/{dataset_id}/metadata/{metadata_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_metadata(
    dataset_id: str = Path(..., description="知识库ID"),
    metadata_id: str = Path(..., description="元数据ID"),
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    删除知识库元数据
    
    Args:
        dataset_id: 知识库ID
        metadata_id: 元数据ID
    """
    try:
        dify_client.delete_dataset_metadata(
            dataset_id=dataset_id,
            metadata_id=metadata_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除元数据失败: {str(e)}"
        )

@router.get("/datasets", response_model=List[DatasetResponse])
async def list_datasets(
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    获取知识库列表
    
    Returns:
        List[DatasetResponse]: 知识库列表
    """
    try:
        result = dify_client.get_dataset_list()
        return result.get("data", [])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取知识库列表失败: {str(e)}"
        )

@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset_detail(
    dataset_id: str = Path(..., description="知识库ID"),
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    获取知识库详情
    
    Args:
        dataset_id: 知识库ID
        
    Returns:
        DatasetResponse: 知识库详情
    """
    try:
        result = dify_client.get_dataset_detail(dataset_id=dataset_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取知识库详情失败: {str(e)}"
        )

@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset: DatasetCreate,
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    创建知识库
    
    Args:
        dataset: 知识库创建请求
        
    Returns:
        DatasetResponse: 创建的知识库
    """
    try:
        result = dify_client.create_dataset(
            name=dataset.name,
            description=dataset.description,
            indexing_technique=dataset.indexing_technique,
            permission=dataset.permission
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建知识库失败: {str(e)}"
        )

@router.patch("/datasets/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str = Path(..., description="知识库ID"),
    dataset: DatasetUpdate = Field(...),
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    更新知识库
    
    Args:
        dataset_id: 知识库ID
        dataset: 知识库更新请求
        
    Returns:
        DatasetResponse: 更新后的知识库
    """
    try:
        # 构建更新参数
        update_params = {}
        if dataset.name is not None:
            update_params["name"] = dataset.name
        if dataset.description is not None:
            update_params["description"] = dataset.description
        if dataset.indexing_technique is not None:
            update_params["indexing_technique"] = dataset.indexing_technique
        if dataset.permission is not None:
            update_params["permission"] = dataset.permission
            
        result = dify_client.update_dataset(
            dataset_id=dataset_id,
            **update_params
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新知识库失败: {str(e)}"
        )

@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str = Path(..., description="知识库ID"),
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    删除知识库
    
    Args:
        dataset_id: 知识库ID
    """
    try:
        dify_client.delete_dataset(dataset_id=dataset_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除知识库失败: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/index/status")
async def get_dataset_index_status(
    dataset_id: str = Path(..., description="知识库ID"),
    dify_client: DifySDKAdapter = Depends(get_dify_client)
):
    """
    获取知识库索引状态
    
    Args:
        dataset_id: 知识库ID
        
    Returns:
        索引状态信息
    """
    try:
        result = dify_client.get_dataset_index_status(dataset_id=dataset_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取索引状态失败: {str(e)}"
        )
