OMP课程2，BiliBili链接：https://www.bilibili.com/video/BV1NQAKzXEC1?spm_id_from=333.788.videopod.sections&vd_source=66786f67c0ee54aa9e72f813720ae1bf

spmd：单程序，多数据

#pragma omp for schedule(static,20) //静态调度
#pragma omp for schedule(dynamic,20) //动态调度