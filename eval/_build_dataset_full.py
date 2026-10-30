# -*- coding: utf-8 -*-
"""生成医学标注集全量：dataset_medical_full.json（100 题 = 第一批 40 核心 + 第二批 60 扩展）。

第二批设计口径：
- 四篇文档各 15 题，重点补第一批题型缺口：中医的多选/问答题/论述/第七章、传染病的简答、
  推拿的单选/多选/论述、西医的填空/简答；
- query 为改写问句，不与原文题干重合；
- must_contain 必须是被标注块内的原文子串（run_eval 按子串匹配）；
- expect_section 取解析器产出的章节路径（子串匹配即可命中）。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.parsing import parse_document  # noqa: E402

EVAL = Path(__file__).resolve().parent
CORPUS = EVAL / "corpus_medical"

# ---------- 第一批（读自 dataset_medical.json，保持不动） ----------
FIRST = json.loads((EVAL / "dataset_medical.json").read_text(encoding="utf-8"))["questions"]

# ---------- 第二批 60 题 ----------
SECOND = [
    # ===== 中医内科习题集（m41-m55）=====
    {"id": "m41", "question": "胁痛之实证主要以哪些病因为主？三者中又以什么为先？", "doc": "中医内科习题集", "must_contain": ["气滞、血瘀、湿热为主"], "expect_section": "第四章  肝胆病证 > 一、填空题："},
    {"id": "m42", "question": "黄疸的证候一般以何种表现先出现？", "doc": "中医内科习题集", "must_contain": ["两目先黄"], "expect_section": "第四章  肝胆病证 > 一、填空题："},
    {"id": "m43", "question": "内伤咳嗽多属邪实正虚，治疗上宜采用哪些方法？", "doc": "中医内科习题集", "must_contain": ["祛邪止咳"], "expect_section": "第一章 肺系病证 > 三、多项选择题："},
    {"id": "m44", "question": "引起哮病发作的常见诱因有哪些？", "doc": "中医内科习题集", "must_contain": ["气候突变"], "expect_section": "第一章 肺系病证 > 三、多项选择题："},
    {"id": "m45", "question": "胃阴不足型呃逆有哪些主症？", "doc": "中医内科习题集", "must_contain": ["呃声急促而不连续"], "expect_section": "第三章 脾胃肠病证 > 三、多项选择题："},
    {"id": "m46", "question": "消渴病常见的并发症有哪些？", "doc": "中医内科习题集", "must_contain": ["白内障、雀目、耳聋"], "expect_section": "第六章气血津液病证 > 三、多项选择题："},
    {"id": "m47", "question": "水肿的常用治法包括哪些？", "doc": "中医内科习题集", "must_contain": ["C.攻逐"], "expect_section": "第五章肾系病证 > 三、多项选择题："},
    {"id": "m48", "question": "喘而胸满闷窒、咳嗽痰多粘腻色白的喘证应如何辨证治疗？", "doc": "中医内科习题集", "must_contain": ["二陈汤合三子养亲汤"], "expect_section": "第一章 肺系病证 > 四、问答题："},
    {"id": "m49", "question": "虽有便意但临厕努挣乏力、挣则汗出而大便并不干硬的便秘应如何治疗？", "doc": "中医内科习题集", "must_contain": ["黄芪汤"], "expect_section": "第三章 脾胃肠病证 > 四、问答题："},
    {"id": "m50", "question": "咽中如有物梗塞、吞之不下咯之不出的病证，其治法与代表方药是什么？", "doc": "中医内科习题集", "must_contain": ["半夏厚朴汤"], "expect_section": "第六章气血津液病证 > 四、问答题"},
    {"id": "m51", "question": "中医认为胁痛的病因病机可从哪些方面论述？", "doc": "中医内科习题集", "must_contain": ["情志抑郁，或暴怒伤肝"], "expect_section": "第四章  肝胆病证 > 四、论述题"},
    {"id": "m52", "question": "中医认为湿邪在黄疸发病过程中有何重要意义？", "doc": "中医内科习题集", "must_contain": ["黄疸的病机关健是湿"], "expect_section": "第四章  肝胆病证 > 四、论述题"},
    {"id": "m53", "question": "外感头痛与内伤头痛在发病特征上有何不同？", "doc": "中医内科习题集", "must_contain": ["其痛如破"], "expect_section": "第七章  经络肢体病证 > 四、简答题"},
    {"id": "m54", "question": "中医认为痉病的标本关系如何？", "doc": "中医内科习题集", "must_contain": ["以督脉为本，以筋脉为标"], "expect_section": "第七章  经络肢体病证 > 填空题："},
    {"id": "m55", "question": "「治痿独取阳明」之说最早出自哪部经典？", "doc": "中医内科习题集", "must_contain": ["《素问·痿论》"], "expect_section": "第七章  经络肢体病证 > 填空题："},

    # ===== 传染病题库（m56-m70）=====
    {"id": "m56", "question": "目前已确定的人类病毒性肝炎有哪些类型？", "doc": "传染病题库", "must_contain": ["甲肝、乙肝、丙肝、丁肝、戊肝"], "expect_section": "填空题"},
    {"id": "m57", "question": "早期诊断甲型病毒性肝炎应测定血清中的哪种抗体？", "doc": "传染病题库", "must_contain": ["抗HAVIgM"], "expect_section": "填空题"},
    {"id": "m58", "question": "反映乙型肝炎病毒复制及传染性最直接、特异、敏感的指标是什么？", "doc": "传染病题库", "must_contain": ["HBV DNA"], "expect_section": "填空题"},
    {"id": "m59", "question": "流行性出血热在临床上的主要特征是什么？", "doc": "传染病题库", "must_contain": ["发热、休克、充血出血"], "expect_section": "填空题"},
    {"id": "m60", "question": "乙脑患者致死的最主要原因是什么？", "doc": "传染病题库", "must_contain": ["呼吸衰竭"], "expect_section": "填空题"},
    {"id": "m61", "question": "从感染艾滋病病毒到抗体可测出之间的时期称为什么？", "doc": "传染病题库", "must_contain": ["窗口期"], "expect_section": "填空题"},
    {"id": "m62", "question": "治疗霍乱的关键措施是什么？", "doc": "传染病题库", "must_contain": ["补充液体和电解质"], "expect_section": "填空题"},
    {"id": "m63", "question": "霍乱的发病机制主要是什么引起的分泌性腹泻？", "doc": "传染病题库", "must_contain": ["霍乱肠毒素"], "expect_section": "填空题"},
    {"id": "m64", "question": "确诊细菌性痢疾最可靠的依据是什么？", "doc": "传染病题库", "must_contain": ["大便培养出痢疾杆菌"], "expect_section": "填空题"},
    {"id": "m65", "question": "钩体病病原治疗首选的抗生素是什么？", "doc": "传染病题库", "must_contain": ["首选抗生素是青霉素"], "expect_section": "填空题"},
    {"id": "m66", "question": "病毒性肝炎是一种什么性质的疾病？", "doc": "传染病题库", "must_contain": ["以肝脏损害为主"], "expect_section": "名词解释"},
    {"id": "m67", "question": "流行病学中的「传染源」是指什么？", "doc": "传染病题库", "must_contain": ["能将其排出体外的人和动物"], "expect_section": "名词解释"},
    {"id": "m68", "question": "什么叫不规则热？", "doc": "传染病题库", "must_contain": ["体温曲线无一定规律"], "expect_section": "名词解释"},
    {"id": "m69", "question": "霍乱属于国际哪一类传染病？", "doc": "传染病题库", "must_contain": ["国际检疫传染病"], "expect_section": "名词解释"},
    {"id": "m70", "question": "流行性出血热临床上的「三痛」「三红」各指什么？", "doc": "传染病题库", "must_contain": ["头痛、腰痛和眼眶痛"], "expect_section": "简答题"},

    # ===== 推拿题库（m71-m85）=====
    {"id": "m71", "question": "医籍名言「夫法之所施，使患者不知其苦，方称为手法也」出自哪部著作？", "doc": "推拿题库", "must_contain": ["医宗金鉴"], "expect_section": "推拿治疗学 > 二、单项选择"},
    {"id": "m72", "question": "「按摩」一词是在哪个朝代正式改称为「推拿」的？", "doc": "推拿题库", "must_contain": ["明代"], "expect_section": "推拿治疗学 > 二、单项选择"},
    {"id": "m73", "question": "我国现存最早的伤科专著是哪一部？", "doc": "推拿题库", "must_contain": ["仙授理伤续断秘方"], "expect_section": "推拿治疗学 > 二、单项选择"},
    {"id": "m74", "question": "推拿辨证施治的根本原则是什么？", "doc": "推拿题库", "must_contain": ["治病必求其本"], "expect_section": "推拿治疗学 > 二、单项选择"},
    {"id": "m75", "question": "挺腹试验阳性提示哪个部位的神经根受压？", "doc": "推拿题库", "must_contain": ["腰部神经根受压"], "expect_section": "推拿治疗学 > 二、单项选择"},
    {"id": "m76", "question": "出现「方肩」畸形时应首先考虑什么？", "doc": "推拿题库", "must_contain": ["肩关节脱位"], "expect_section": "推拿治疗学 > 二、单项选择"},
    {"id": "m77", "question": "推拿治疗的基本作用包括哪些方面？", "doc": "推拿题库", "must_contain": ["A.调整脏腑 B.疏通经络"], "expect_section": "推拿治疗学 > 三、多选题"},
    {"id": "m78", "question": "屈腕试验阳性可提示哪些情况？", "doc": "推拿题库", "must_contain": ["正中神经受压"], "expect_section": "推拿治疗学 > 三、多选题"},
    {"id": "m79", "question": "临床上常见的颈椎病分型有哪些？", "doc": "推拿题库", "must_contain": ["神经根型颈椎病"], "expect_section": "推拿治疗学 > 三、多选题"},
    {"id": "m80", "question": "网球肘的病理本质和别称是什么？", "doc": "推拿题库", "must_contain": ["肱骨外上髁炎"], "expect_section": "推拿治疗学 > 四、名词解释"},
    {"id": "m81", "question": "「漏肩风」在临床上指什么病？", "doc": "推拿题库", "must_contain": ["肩关节周围炎的别称"], "expect_section": "推拿治疗学 > 四、名词解释"},
    {"id": "m82", "question": "腕管综合征的发病机制是什么？", "doc": "推拿题库", "must_contain": ["正中神经在腕管内受到压迫"], "expect_section": "推拿治疗学 > 四、名词解释"},
    {"id": "m83", "question": "落枕的典型症状有哪些？", "doc": "推拿题库", "must_contain": ["急性颈项部肌肉痉挛"], "expect_section": "推拿治疗学 > 四、名词解释"},
    {"id": "m84", "question": "肘后三角由哪些骨性标志构成？", "doc": "推拿题库", "must_contain": ["肱骨内、外上髁和尺骨鹰咀"], "expect_section": "推拿治疗学 > 四、名词解释"},
    {"id": "m85", "question": "腰椎间盘突出症的推拿治疗原则是什么？", "doc": "推拿题库", "must_contain": ["舒筋通络，活血止痛，理筋整复"], "expect_section": "推拿治疗学 > 六、论述题（10分）"},

    # ===== 西医内科（m86-m100）=====
    {"id": "m86", "question": "什么是肺性脑病？它为什么是肺心病死亡的首要原因？", "doc": "西医内科", "must_contain": ["精神障碍、神经症状"], "expect_section": "名词解释："},
    {"id": "m87", "question": "支气管哮喘的病理本质是什么？由哪些炎症细胞介导？", "doc": "西医内科", "must_contain": ["肥大细胞、嗜酸性粒细胞"], "expect_section": "名词解释："},
    {"id": "m88", "question": "心源性哮喘的另一个名称是什么？", "doc": "西医内科", "must_contain": ["夜间阵发性呼吸困难"], "expect_section": "名词解释："},
    {"id": "m89", "question": "高血压脑病发作时血压的哪一项突然升高？", "doc": "西医内科", "must_contain": ["舒张压突然升高"], "expect_section": "名词解释："},
    {"id": "m90", "question": "肝硬化的发病本质是什么？", "doc": "西医内科", "must_contain": ["长期损害肝脏"], "expect_section": "名词解释："},
    {"id": "m91", "question": "糖尿病的核心特征是什么？", "doc": "西医内科", "must_contain": ["慢性高血糖"], "expect_section": "名词解释："},
    {"id": "m92", "question": "支气管哮喘最典型的临床表现是什么？", "doc": "西医内科", "must_contain": ["反复发作性的呼气性呼吸困难"], "expect_section": "填空题："},
    {"id": "m93", "question": "肺炎球菌肺炎的首选治疗药物是什么？", "doc": "西医内科", "must_contain": ["肺炎球菌肺炎首选青霉素G"], "expect_section": "填空题："},
    {"id": "m94", "question": "确诊肺结核最特异的检查是什么？", "doc": "西医内科", "must_contain": ["结核菌检查"], "expect_section": "填空题："},
    {"id": "m95", "question": "什么脉搏体征提示左心功能不全？", "doc": "西医内科", "must_contain": ["交替脉"], "expect_section": "填空题："},
    {"id": "m96", "question": "按 1999 年 WHO/ISH 标准，高血压的诊断界值是什么？", "doc": "西医内科", "must_contain": ["收缩压≥140mmHg"], "expect_section": "填空题："},
    {"id": "m97", "question": "对冠心病具有确诊价值的检查是什么？", "doc": "西医内科", "must_contain": ["冠状动脉造影"], "expect_section": "填空题："},
    {"id": "m98", "question": "慢性支气管炎的诊断标准是什么？", "doc": "西医内科", "must_contain": ["连续两年或以上"], "expect_section": "简答题："},
    {"id": "m99", "question": "慢性肺源性心脏病可能出现哪些并发症？", "doc": "西医内科", "must_contain": ["酸碱平衡失调及电解质紊乱"], "expect_section": "简答题："},
    {"id": "m100", "question": "NYHA 心功能分级中，Ⅳ级患者的主要特征是什么？", "doc": "西医内科", "must_contain": ["不能从事任何体力活动"], "expect_section": "简答题："},
]

DATASET = {
    "description": "医学领域真实题库外部验证集（全量 100 题 = 核心 40 + 扩展 60）：中医内科习题集 / 传染病题库 / 推拿题库 / 西医内科 各 25 题。语料来自真实医学题库文档（eval/corpus_medical/）。query 为改写问句，must_contain 为答案块原文子串，expect_section 为解析器产出章节路径。",
    "note": "第一批 40 题核心集优先名词解释/简答/论述；第二批 60 题扩展集补齐多选/单选/填空/问答/简答题型与第四章、第七章、推拿特殊检查、西医消化/泌尿等覆盖面。",
    "questions": FIRST + SECOND,
}


def _check_must_contain() -> list[str]:
    texts: dict[str, str] = {}
    for f in sorted(CORPUS.glob("*.md")):
        res = parse_document(f.name, str(CORPUS / f.name))
        texts[f.stem] = "\n".join(b.text for b in res.blocks)
    problems = []
    for q in DATASET["questions"]:
        if q["doc"] not in texts:
            problems.append(f"{q['id']}: 找不到文档 {q['doc']}")
            continue
        for kw in q["must_contain"]:
            if kw not in texts[q["doc"]]:
                problems.append(f"{q['id']}: 关键词不在原文中 -> {kw}")
    return problems


def main() -> None:
    qs = DATASET["questions"]
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        dup = {i for i in ids if ids.count(i) > 1}
        print(f"id 重复: {dup}")
        sys.exit(1)
    problems = _check_must_contain()
    if problems:
        print("校验失败：")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    docs = {}
    for q in qs:
        docs[q["doc"]] = docs.get(q["doc"], 0) + 1
    out = EVAL / "dataset_medical_full.json"
    out.write_text(json.dumps(DATASET, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"生成完成：{out.name} 共 {len(qs)} 题；按文档分布 {docs}；全部 must_contain 已在原文中核验通过。")


if __name__ == "__main__":
    main()
