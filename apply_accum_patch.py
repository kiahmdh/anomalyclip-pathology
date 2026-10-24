import sys, shutil, os

path = sys.argv[1] if len(sys.argv) > 1 else 'train.py'
src = open(path).read()

if 'accum_steps' in src:
    print('Already patched (found accum_steps). Nothing to do.')
    sys.exit(0)

old_for = "        for items in tqdm(train_dataloader):"
new_for = ("        accum_steps = 4  # micro-batch 16 x 4 = effective batch 64\n"
           "        optimizer.zero_grad()\n"
           "        for step, items in enumerate(tqdm(train_dataloader)):")

old_tail = ("            loss = lam * loss\n"
            "            optimizer.zero_grad()\n"
            "            (loss+image_loss).backward()\n"
            "            optimizer.step()\n"
            "            loss_list.append(loss.item())")
new_tail = ("            loss = lam * loss\n"
            "            (loss + image_loss).div(accum_steps).backward()\n"
            "            if (step + 1) % accum_steps == 0:\n"
            "                optimizer.step()\n"
            "                optimizer.zero_grad()\n"
            "            loss_list.append(loss.item())")

for label, old in [('for-loop line', old_for), ('loss/step block', old_tail)]:
    n = src.count(old)
    if n != 1:
        print(f'ABORT: expected to find the {label} exactly once, found {n} times.')
        print('No changes made. Your train.py is untouched.')
        sys.exit(1)

shutil.copy(path, path + '.bak')
src = src.replace(old_for, new_for).replace(old_tail, new_tail)
open(path, 'w').write(src)
print(f'Patched {path}. Backup saved as {path}.bak')
print('accum_steps = 4  ->  run with --batch_size 16  (effective batch 64)')
