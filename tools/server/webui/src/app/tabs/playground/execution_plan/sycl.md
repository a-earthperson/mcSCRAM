The index space of an ND-Range kernel is divided into work-groups, sub-groups, and work-items. A work-item is the basic unit. A collection of work-items form a sub-group, and a collection of sub-groups form a work-group. The mapping of work-items and work-groups to hardware vector engines (VE) is implementation-dependent. All the work-groups run concurrently but may be scheduled to run at different times depending on availability of resources. Work-group execution may or or may not be preempted depending on the capabilities of underlying hardware. Work-items in the same work-group are guaranteed to run concurrently. Work-items in the same sub-group may have additional scheduling guarantees and have access to additional functionality.

A sub-group is a collection of contiguous work-items in the global index space that execute in the same VE thread. When the device compiler compiles the kernel, multiple work-items are packed into a sub-group by vectorization so the generated SIMD instruction stream can perform tasks of multiple work-items simultaneously. Properly partitioning work-items into sub-groups can make a big performance difference.


GPU Execution Model and Mapping

The SYCL execution model exposes an abstract view of GPU execution. The SYCL thread hierarchy consists of a 1-, 2-, or 3-dimensional grid of work-items. A collection of work-items forms a sub-group, and a collection of sub-groups forms a work-group (see the following illustration).

Each work-item is mapped to SIMD lane.
Sub-group is mapped to Vector Engine hardware thread.
Work-group is executed in a Xe-Core.

## Work-item

A work-item represents one of a collection of parallel executions of a kernel.

## Sub-group

A sub-group represents a short range of consecutive work-items that are processed together as a SIMD vector of length 16 or 32.

## Work-group

A work-group is a 1-, 2-, or 3-dimensional representation of a collection of related work-items that are executed on a single Xe-Core. In SYCL, synchronization across work-items is only possible with barriers for the work-items within the same work-group.

## nd_range

An nd_range divides the thread hierarchy into 1-, 2-, or 3-dimensional grids of work-groups. It is represented by the global range, the local range of each work-group.