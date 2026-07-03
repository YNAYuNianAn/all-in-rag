from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf


#pdf 文档路径
pdf_path = "../../data/C2/pdf/rag.pdf"

#使用Unstructrued 加载并解析PDF
elements = partition(
    filename = pdf_path,
    content_type = "application/pdf"
)
elements1 = partition_pdf(
    filename = pdf_path,
    strategy = "hi_res",
    extract_images_in_pdf=True
)
# print(type(elements))

#打印解析结果
# print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements) }")
print(f"解析完成: {len(elements1)} 个元素, {sum(len(str(e)) for e in elements1) }")
#统计元素类型
from collections import Counter
# types = Counter(e.category for e in elements)
# print(f"元素类型: {dict(types)}")
types = Counter(e.category for e in elements1)
print(f"元素类型: {dict(types)}")
#显示所有元素
print("\n所有元素:")
# for i,element in enumerate(elements,1) :
#     print(f"Element {i} ({element.category})")
#     print(element)
#     print("=" * 60)
for i,element in enumerate(elements1,1) :
    print(f"Element {i} ({element.category})")
    print(element)
    print("=" * 60)